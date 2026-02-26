# Version History

---

All notable changes to this project will be documented in this file.

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
