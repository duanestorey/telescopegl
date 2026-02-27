# Version History

---

All notable changes to this project will be documented in this file.

## [2.1.0](https://github.com/duanestorey/polymerase/tree/main) — 2026-02-27

Comprehensive code review: 43 fixes across 22 files addressing resource safety,
correctness, dead code, and architectural standardization.

### Fixed

- **Resource safety**: BAM file handles in `_load_sequential` wrapped in
  try/finally. `Pool` in `_load_parallel` uses context manager. File handles
  in `_mapping_fromfiles`, `fetch_region`, and GTF loading all use proper
  context managers or try/finally.

- **Operator precedence bug in `get_random_seed`**: `% shape[0] * shape[1]`
  was `(% shape[0]) * shape[1]` due to Python precedence; fixed to
  `% (shape[0] * shape[1])`.

- **`__str__` on Polymerase**: Was returning unformatted `%s` placeholders;
  now uses f-strings with actual attribute values.

- **`em_parallel` tracking**: `pi_init` now reconstructed from per-block
  initial values (was just a copy of final pi). `num_iterations` is now the
  max across blocks (was hardcoded to `max_iter`).

- **`update_sam` KeyError**: `read_index[...]` replaced with `.get()` + skip
  to handle fragments present in tmp BAM but filtered from the matrix.

- **Stub overlap modes**: `intersection-strict` and `union` now raise
  `NotImplementedError` instead of silently returning `None`.

- **Numba `threshold_filter`/`indicator`**: Added missing `eliminate_zeros()`
  calls, matching the `binmax`/`choose_random` pattern.

- **`np.seterr` leak**: `_recip0` now uses `np.errstate` context manager.

- **`np.random.choice(range(...))`**: Replaced with `np.random.randint()` in
  `choose_random` (avoids materializing a Python list).

- **`find_blocks` memory**: Binary adjacency matrix uses `np.bool_` instead
  of `np.float64` (8x smaller).

- **`split_matrix` type preservation**: Uses `type(score_matrix)(sub)` to
  preserve `CpuOptimizedCsrMatrix` through block decomposition.

- **GPU `_to_cpu`**: Now checks `cpsparse.issparse()` first, correctly
  transferring CuPy sparse matrices back to scipy. Fixes `multiply`, `dot`,
  and `sum` on GPU backend.

- **GPU `to_sparse`**: Returns CuPy sparse on GPU backend instead of scipy.

- **GPU array utilities**: `zeros`, `array`, `arange` no longer immediately
  `.get()` results back to CPU, keeping data on GPU as intended.

- **`fetch_bundle` StopIteration**: Catches `StopIteration` on empty BAM
  instead of propagating.

- **`intervaltree` save/load**: `run_stranded` now persisted in pickle.

- **`bisect.py` `feature_length`**: Fixed incorrect `self._locus[locus_idx]`
  lookup — `locus_idx` was already the name string, not an index.

- **`bisect.py` `lookup` assert**: Replaced `assert len(possible) == 1` with
  proper `ValueError`.

- **`annotation/__init__.py` error message**: Now shows the actual invalid
  class name and states only "intervaltree" is supported.

- **`fetch_region` assert**: Replaced `assert CODES[ci][0] == 'PX*'` with
  warning log.

- **AssignPrimer version**: Now imports `__version__` from package instead of
  hardcoding.

- **`yaml.FullLoader`**: Replaced with `yaml.SafeLoader` in CLI option parsing.

### Added

- **Auto-download default annotation**: When no `--gtffile` is specified and
  the bundled GTF is not present (e.g., pip install), automatically downloads
  `retro.hg38.v1` from `mlbendall/telescope_annotation_db` to
  `~/.polymerase/annotations/`. Cached for subsequent runs.

- **`choose_random` and `indicator` abstract methods** on `ComputeOps` ABC,
  with implementations in both `CpuOps` and `GpuOps`.

- **Package name validation** in `polymerase install` — rejects names that
  don't match PEP 508 format before invoking pip.

- **`exc_info=True`** on all registry warning handlers for better debugging.

- **Shared CLI utilities**: `backend_display_name()` and
  `collect_output_files()` moved from `assign.py` to `cli/__init__.py`,
  eliminating cross-module imports in `resume.py`.

### Removed

- Dead `_norm_loop` method from `csr_matrix_plus`.
- Dead `save_memory` parameter from `em()`.
- Dead `_known_cols` set and `_parts` assignment in normalize cofactor.
- Dead `if scipy.sparse.issparse(m): pass` in GPU `nmf`.
- Dead if/else branch in `SubcommandOptions.__init__`.
- `cython` from runtime dependencies (kept in build-system.requires).

### Changed

- `_resolve_duplicates_max` uses `np.maximum.reduceat` instead of Python loop.
- Version bumped to 2.1.0 in `__init__.py` and `pyproject.toml`.

---

## [2.0.3](https://github.com/duanestorey/polymerase/tree/main) — 2026-02-26

Production pipeline fixes from testing against full RepeatMasker GTFs and
real RNA-seq data (SRR9666161, 14.7M fragments).

### Fixed

- **family-agg attribute auto-detection**: The `family-agg` cofactor now
  auto-detects whether the GTF uses `repFamily`/`repClass` (retro.hg38 format)
  or `family_id`/`class_id` (RepeatMasker standard format). Previously all
  families mapped to "Unknown" with RepeatMasker GTFs. Detection logic lives in
  the new `annotation/gtf_utils.py` module.

- **Interval merging assertion crash**: Replaced `assert len(mergeable) == 1`
  in `intervaltree.py` with a loop that merges all overlapping intervals sharing
  the same attribute key. This fixes crashes when using `--attribute gene_id`
  with GTFs containing nested TE insertions. Also removed vestigial `if True:`.

- **run_stats.tsv header format**: Added missing newline after the RunInfo
  comment line. The file now parses cleanly with
  `pd.read_csv(..., comment='#')`.

- **`--logfile` write mode**: Changed `argparse.FileType('r')` to
  `argparse.FileType('w')` in all four option classes (BulkIDOptions,
  scIDOptions, BulkResumeOptions, scResumeOptions).

### Added

- **`--classes` filter**: New CLI option for `polymerase assign` that accepts a
  comma-separated list of TE classes (e.g., `--classes LTR,LINE,SINE`). Filters
  GTF exon lines during loading, enabling use of full 4.8M-entry RepeatMasker
  GTFs that would otherwise exhaust memory. Auto-detects the class attribute
  name (`repClass` or `class_id`).

- **Performance Optimization section** in README: Documents optional
  acceleration packages (`numba`, `sparse-dot-mkl`, `cupy`) with a backend
  comparison table.

- **Version History section** in README.

- **BAI pre-creation hint**: Informational log message when auto-creating a
  `.bai` index, suggesting pipelines pre-create it with `samtools index`.

- **`--outdir` info message**: Logs a suggestion to use `--outdir` when output
  defaults to the current directory.

### Changed

- **Skip SAM tag writing on indexed path**: `process_overlap_frag()` now
  accepts `write_tags=False` to skip ZF/ZT/ZB tag writes when `--updated_sam`
  is not requested. Reduces per-fragment processing overhead on the indexed
  loading path.

- Version bumped to 2.0.3 in `__init__.py` and `pyproject.toml`.

---

## [2.0.2](https://github.com/duanestorey/polymerase/tree/main) — 2026-02-26

### Added

- **Ruff linting and formatting** configuration in `pyproject.toml`.
- **GitHub Actions CI** (`.github/workflows/test.yml`): lint job with
  `astral-sh/ruff-action` plus conda-based test matrix (Python 3.10, 3.12,
  3.13 on ubuntu-latest).

### Changed

- Applied `ruff check --fix` and `ruff format` across 49 files (net -56 lines
  of dead code removed).

---

## [2.0.1](https://github.com/duanestorey/polymerase/tree/main) — 2026-02-26

### Added

- **Benchmark timing table**: A timing summary is displayed at the end of every
  `assign` and `resume` run, showing per-stage elapsed time and percentage.
  Verbose mode (`--verbose`) adds cofactor sub-stage detail. Implemented via the
  `Stopwatch` class and `Console.timing_table()` in `cli/console.py`.

- **`stopwatch` parameter on plugin ABCs**: `Primer.commit()` and
  `Cofactor.transform()` accept an optional `stopwatch` kwarg, allowing the
  platform to collect fine-grained timing from plugins without coupling.

### Fixed

- **`--outdir` auto-creation**: The pipeline now creates the output directory
  if it doesn't exist, instead of crashing with `FileNotFoundError` when
  saving the checkpoint. Fixed in both `assign` and `resume` subcommands.

---

## [2.0.0](https://github.com/duanestorey/polymerase/tree/main) — 2026-02-26

### Added

- **Indexed BAM loading** (`_load_indexed` in `model.py`): Two-pass approach for
  coordinate-sorted indexed BAMs. Phase 1 fetches TE regions to identify relevant
  fragments, Phase 2 buffers all alignments for those fragments, Phase 3 processes
  identically to the sequential path. Produces exact matrix match with 4-8x speedup.
  Auto-detected when BAM has a `.bai` index.

- **Backend abstraction** (`backend.py`): Auto-detects available acceleration
  libraries. CPU-Optimized/Numba+MKL -> Stock CPU/scipy. Configured via
  `backend.configure()`, accessed via `get_backend()` and `get_sparse_class()`.

- **Numba JIT sparse kernels** (`numba_matrix.py`):
  `CpuOptimizedCsrMatrix` subclass of `csr_matrix_plus` with `@numba.njit`
  kernels for `binmax`, `choose_random`, `threshold_filter`, `indicator`.
  Falls back to stock scipy when Numba not installed.

- **Connected component block decomposition** (`decompose.py`): Decomposes the
  N x K score matrix into independent sub-problems via
  `scipy.sparse.csgraph.connected_components`. Functions: `find_blocks()`,
  `split_matrix()`, `merge_results()`.

- **Block-parallel EM** (`em_parallel` in `likelihood.py`): Runs EM independently
  on each connected component block using `ThreadPoolExecutor`. Falls back to
  standard EM when only 1 block exists. Produces identical results to full-matrix EM.

- **Batch COO matrix construction** (`model.py`): Replaced sequential DOK loop with
  bulk COO array collection and `_resolve_duplicates_max()` for efficient sparse
  matrix building.

- **pysam multi-threaded decompression**: All `pysam.AlignmentFile` opens now use
  `threads=N` for parallel BAM decompression.

- **`threshold_filter()` and `indicator()` methods** on `csr_matrix_plus`: Named
  methods replacing anonymous lambdas in reassignment, enabling Numba acceleration.

- **1% test data** (`test_data/`): `SRR9666161_1pct_collated.bam` (30MB) and
  `retro.hg38.gtf` (18MB) checked into the repository for CI-friendly pipeline tests.

- **211 tests** across 9 test files covering sparse ops, EM algorithm, block
  decomposition, CPU-optimized equivalence, indexed path equivalence,
  full-scale pipeline benchmarking, compute abstraction, and plugin framework.

- `CHANGELOG.md` for documenting changes

- `DEVELOPER.md` developer notes

- `CLAUDE.md` project context for Claude Code

- **Plugin architecture** (`polymerase/plugins/`): Platform loads BAM/GTF once,
  then delegates analysis to discoverable **primers** (analysis plugins) and
  **cofactors** (post-processors). Plugins are discovered via Python
  `entry_points` in `pyproject.toml`.

- **`assign` primer** (`plugins/builtin/assign.py`): The original EM pipeline
  refactored as the first built-in primer. Receives annotation and alignment
  snapshots via platform hooks, runs EM, and writes counts/stats/updated SAM.

- **`family-agg` cofactor** (`plugins/builtin/family_agg.py`): Aggregates
  `assign` counts by `repFamily` and `repClass` from the GTF annotation.
  Outputs `*-family_counts.tsv` and `*-class_counts.tsv`.

- **`normalize` cofactor** (`plugins/builtin/normalize.py`): Computes TPM,
  RPKM, and CPM from `assign` counts and feature lengths. Outputs
  `*-normalized_counts.tsv`.

- **Compute abstraction layer** (`polymerase/compute/`): Backend-agnostic math
  operations (`ComputeOps` ABC) with `CpuOps` (scipy/numpy) and `GpuOps`
  (CuPy/cuSPARSE) implementations. Primers use `ops.dot()`, `ops.norm()`, etc.
  and get GPU acceleration automatically when available.

- **Plugin management CLI**: `polymerase list-plugins` lists all installed
  primers and cofactors. `polymerase install <package>` installs third-party
  plugin packages.

- **`pyproject.toml`**: Modern build configuration with entry_points for plugin
  discovery. Replaces versioneer for version management.

- **Frozen snapshot dataclasses** (`plugins/snapshots.py`): `AnnotationSnapshot`
  and `AlignmentSnapshot` provide immutable data to primers via platform hooks.

- **Plugin registry** (`plugins/registry.py`): Discovers, configures, notifies,
  and commits all active primers and cofactors. Failing plugins log warnings
  without crashing the pipeline.

### Changed

- **Renamed project from Telescope to Polymerase**: Full package restructure into
  extensible subpackages (`cli/`, `core/`, `sparse/`, `annotation/`, `alignment/`,
  `utils/`). Classes renamed: `Telescope` → `Polymerase`, `scTelescope` →
  `scPolymerase`, `TelescopeLikelihood` → `PolymeraseLikelihood`. CLI commands
  are now `polymerase assign` and `polymerase resume`. Attribution to the original
  Telescope project by Matthew Bendall is preserved throughout.

- **Coordinate-sorted BAMs now supported**: Previously required collated/name-sorted
  BAMs. Indexed BAMs are now auto-detected and use the faster region-based loading
  path.

- **Module split**: `model.py` (833 lines) split into `model.py` (Polymerase
  orchestrator + BAM loading), `likelihood.py` (EM algorithm), `reporter.py`
  (output I/O). Backward-compatible re-exports maintained.

- **`run()` refactored into thin platform orchestrator** (`cli/assign.py`):
  Loads data, builds frozen snapshots, and delegates to `PluginRegistry`.
  All EM logic now lives inside the `assign` primer's `commit()` method.

- **Reporter decoupled from Polymerase object** (`core/reporter.py`):
  `output_report()` and `update_sam()` now accept individual data pieces
  (feat_index, feature_length, run_info, etc.) instead of a Polymerase instance.

- **Backend-aware sparse construction**: `likelihood.py` and `model.py` use
  `get_sparse_class()` from `backend.py` so the correct sparse class
  (`CpuOptimizedCsrMatrix` or `csr_matrix_plus`) is used throughout the pipeline.

- **Backend detection extended for GPU**: `backend.py` now detects CuPy/cuSPARSE
  availability with three-tier detection: GPU > CPU-Optimized > CPU Stock.

- **Removed versioneer**: Replaced with static `__version__ = "2.0.0"` and
  `pyproject.toml` metadata. Deleted `versioneer.py` (2205 lines) and
  `polymerase/_version.py` (659 lines).

- **Security fix**: Replaced `eval()` in CLI option type parsing with safe
  lookup dictionary (`_SAFE_TYPES`).

- **NumPy 2.0 compatibility**: Fixed `np.float` → `np.float64` in `matrix.py`.

- Depends on python >= 3.10 (updated from 3.7).

- Removed `bulk` and `sc` subcommands since the single-cell
  [stellarscope](https://github.com/nixonlab/stellarscope) project has moved
  into its own repository.

- Migrated test framework from nose to pytest.

### Fixed

- **Supplementary alignment classification bug**: `fetch_fragments_seq` used
  `alns[0]` for PM/PX classification, but `alns[0]` depends on BAM sort order.
  Supplementary alignments appearing before primary alignments in collated BAMs
  caused incorrect PX classification. Fixed by adding `_find_primary()` which
  finds the first non-supplementary, non-secondary alignment.

- Error where numpy.int is deprecated

- Cython 3.0 + pysam 0.23.3 compatibility

- Python 3.13 `'rU'` mode deprecation

- `stranded_mode` None vs 'None' comparison

----

_The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)._


## Previous versions

Prior to this CHANGELOG, changes were tracked in [README.md](./README.md).
The previous version history from the original Telescope project is relocated here:

### v1.0.3.1

  + Checks that GTF feature type is "exon"
  + Skips GTF lines missing attribute (with warning)

### v1.0.3
  + Added cimport statements to calignment.pyx (MacOS bug fix)
  + Fixed warning about deprecated PyYAML yaml.load
  + Compatibility with intervaltree v3.0.2

### v1.0.2
  + Temporary files are written as BAM

### v1.0.1
  + Changed default `theta_prior` to 200,000

### v1.0
  + Removed dependency on `git`
  + Release version

### v0.6.5
  + Support for sorted BAM files
  + Parallel reading of sorted BAM files
  + Improved performance of EM
  + Improved memory-efficiency of spare matrix

#### v0.5.4.1
  + Fixed bug where random seed is out of range

#### v0.5.4
  + Added MIT license
  + Changes to logging/reporting

#### v0.5.3
  + Improvements to `polymerase resume`

#### v0.5.2
  + Implemented checkpoint and `polymerase resume`

#### v0.5.1
  + Refactoring Polymerase class with PolymeraseLikelihood
  + Improved memory usage

#### v0.4.2
  +  Subcommand option parsing class
  +  Cython for alignment parsing
  +  HTSeq as alternate alignment parser

#### v0.3.2
  +  Python3 compatibility

#### v0.3
  + Implemented IntervalTree for Annotation data structure
  + Added support for annotation files where a locus may be non-contiguous.
  + Overlapping annotations with the same key value (locus) are merged
  + User can set minimum overlap criteria for assigning read to locus, default = 0.1

#### v0.2

  + Implemented checkpointing
  + Output tables as pickled objects
  + Changes to report format and output
