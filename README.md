Polymerase
========

**Based on the original [Telescope](https://github.com/mlbendall/telescope) project by Matthew Bendall.**

**Original Telescope Affiliations:**

+ [Computational Biology Institute](http://cbi.gwu.edu) at George Washington University
+ [Weill Cornell Medicine Division of Infectious Diseases](https://medicine.weill.cornell.edu/divisions-programs/infectious-diseases)

**Table of Contents:**

* [Installation](#installation)
* [Usage](#usage)
  * [`polymerase assign`](#polymerase-assign)
  * [`polymerase resume`](#polymerase-resume)
  * [`polymerase list-plugins`](#polymerase-list-plugins)
  * [`polymerase install`](#polymerase-install)
* [Plugin Architecture](#plugin-architecture)
  * [Primers](#primers)
  * [Cofactors](#cofactors)
  * [Writing a Plugin](#writing-a-plugin)
* [Output](#Output)
  * [Polymerase report](#polymerase-report)
  * [Updated SAM file](#updated-sam-file)
  * [Cofactor output](#cofactor-output)
* [Version History](#version-history)

## Installation

**Latest version:**

Use conda/mamba package manager to install dependencies, then
use `pip` to install Polymerase.

```bash
mamba env create -n polymerase https://github.com/duanestorey/polymerase/raw/main/environment.yml
conda activate polymerase
pip install git+https://github.com/duanestorey/polymerase.git
polymerase --version
```

## Testing

A BAM file (`alignment.bam`) and annotation (`annotation.gtf`) are included in
the polymerase package for testing. The files are installed in the `data`
directory of the package root. We've included a subcommand, `polymerase test`,
to generate an example command line with the correct paths.
For example, to generate an example command line:

```
polymerase test
```

The command can be executed using `eval`:

```
eval $(polymerase test)
```

The expected output to STDOUT includes the final log-likelihood, which was
`95252.596293` in our tests. The test also outputs a report,
`polymerase-report.tsv`, which can be compared to the report
included in the `data` directory. NOTE: The precise values may be
platform-dependent due to differences in floating point precision.

## Usage

### `polymerase assign`

The `polymerase assign` program finds overlapping reads between an alignment
(SAM/BAM) and an annotation (GTF) then reassigns reads using a statistical
model. This algorithm enables locus-specific quantification of transposable
element expression.

#### Basic usage

Basic usage requires a file containing read alignments to the genome and an
annotation file with the transposable element gene model.

```
polymerase assign [samfile] [gtffile]
```

Polymerase accepts both **coordinate-sorted indexed BAMs** and **collated/name-sorted BAMs**.

For coordinate-sorted BAMs with a `.bai` index, Polymerase automatically uses
region-based loading that only reads TE-overlapping regions, which is 4-8x faster
than a full scan. To create an indexed BAM:

```bash
samtools sort -o aligned_sorted.bam aligned.bam
samtools index aligned_sorted.bam
```

For collated BAMs (where all alignments for a read pair appear sequentially),
the default SAM/BAM output for many aligners is in the correct order, or BAM
files can be sorted by read name (`samtools sort -n`). A faster alternative
to a full read name sort is [`samtools collate`](http://www.htslib.org/doc/samtools-collate.html).

Reads should be aligned and be permitted to map to multiple locations (i.e. `-k` option in `bowtie2`).

The annotation file must be in GTF format and indicate the genomic regions that
represent transposable element transcripts. The transcripts are permitted to be
disjoint in order to exclude insertions of other element types. A collection of
valid transposable element gene models are available for download at
[mlbendall/telescope_annotation_db](https://github.com/mlbendall/telescope_annotation_db).

#### Advanced usage

```
Input Options:

  samfile               Path to alignment file. Alignment file can be in SAM
                        or BAM format. File must be collated so that all
                        alignments for a read pair appear sequentially in the
                        file.
  gtffile               Path to annotation file (GTF format)
  --attribute ATTRIBUTE
                        GTF attribute that defines a transposable element
                        locus. GTF features that share the same value for
                        --attribute will be considered as part of the same
                        locus. (default: locus)
  --no_feature_key NO_FEATURE_KEY
                        Used internally to represent alignments. Must be
                        different from all other feature names. (default:
                        __no_feature)
  --ncpu NCPU           Number of cores to use. (Multiple cores not supported
                        yet). (default: 1)
  --tempdir TEMPDIR     Path to temporary directory. Temporary files will be
                        stored here. Default uses python tempfile package to
                        create the temporary directory. (default: None)

Reporting Options:

  --quiet               Silence (most) output. (default: False)
  --debug               Print debug messages. (default: False)
  --logfile LOGFILE     Log output to this file. (default: None)
  --outdir OUTDIR       Output directory. (default: .)
  --exp_tag EXP_TAG     Experiment tag (default: polymerase)
  --updated_sam         Generate an updated alignment file. (default: False)

  Run Modes:

  --reassign_mode {exclude,choose,average,conf,unique}
                        Reassignment mode. After EM is complete, each fragment
                        is reassigned according to the expected value of its
                        membership weights. The reassignment method is the
                        method for resolving the "best" reassignment for
                        fragments that have multiple possible reassignments.
                        Available modes are: "exclude" - fragments with
                        multiple best assignments are excluded from the final
                        counts; "choose" - the best assignment is randomly
                        chosen from among the set of best assignments;
                        "average" - the fragment is divided evenly among the
                        best assignments; "conf" - only assignments that
                        exceed a certain threshold (see --conf_prob) are
                        accepted; "unique" - only uniquely aligned reads are
                        included. NOTE: Results using all assignment modes are
                        included in the Polymerase report by default. This
                        argument determines what mode will be used for the
                        "final counts" column. (default: exclude)
  --use_every_reassign_mode (single-cell only)
                        Whether to output count matrices using every reassign mode.
                        If specified, six output count matrices will be generated,
                        corresponding to the six possible reassignment methods (all, exclude,
                        choose, average, conf, unique). (default: False)
  --conf_prob CONF_PROB
                        Minimum probability for high confidence assignment.
                        (default: 0.9)
  --overlap_mode {threshold,intersection-strict,union}
                        Overlap mode. The method used to determine whether a
                        fragment overlaps feature. (default: threshold)
  --overlap_threshold OVERLAP_THRESHOLD
                        Fraction of fragment that must be contained within a
                        feature to be assigned to that locus. Ignored if
                        --overlap_method is not "threshold". (default: 0.2)
  --annotation_class {intervaltree,htseq}
                        Annotation class to use for finding overlaps. Both
                        htseq and intervaltree appear to yield identical
                        results. Performance differences are TBD. (default:
                        intervaltree)
  --stranded_mode {None, RF, R, FR, F}
                        Options for considering feature strand when assigning reads.
                        If None, for each feature in the annotation, returns counts
                        for the positive strand and negative strand. If not None,
                        this argument specifies the orientation of paired end reads
                        (RF - read 1 reverse strand, read 2 forward strand) and
                        single end reads (F - forward strand) with respect to the
                        generating transcript. (default: None)
  --barcode_tag (single-cell only)
                        String specifying the name of the field in the BAM/SAM
                        file containing the barcode for each read. (default: CB)
Model Parameters:

  --pi_prior PI_PRIOR   Prior on pi. Equivalent to adding n unique reads.
                        (default: 0)
  --theta_prior THETA_PRIOR
                        Prior on theta. Equivalent to adding n non-unique reads.
                        (default: 200000)
  --em_epsilon EM_EPSILON
                        EM Algorithm Epsilon cutoff (default: 1e-7)
  --max_iter MAX_ITER   EM Algorithm maximum iterations (default: 100)
  --use_likelihood      Use difference in log-likelihood as convergence
                        criteria. (default: False)
  --skip_em             Exits after loading alignment and saving checkpoint
                        file. (default: False)

Performance Options:

  --parallel_blocks     Decompose into independent blocks and solve in
                        parallel. (default: False)
```


### `polymerase list-plugins`

List all installed primers and cofactors:

```
polymerase list-plugins
```

Example output:
```
Primers:
  assign               v2.0.0      Quantify TE expression using Expectation-Maximization (built-in)

Cofactors:
  family-agg           for assign     Aggregate TE counts by repFamily and repClass (built-in)
  normalize            for assign     Compute TPM, RPKM, and CPM normalized counts (built-in)
```

### `polymerase install`

Install a third-party primer or cofactor package:

```
polymerase install polymerase-primer-chimeric
```

This wraps `pip install` and registers the plugin's entry_points so it is
discovered automatically on the next run.

### `polymerase resume`

The `polymerase resume` program loads the checkpoint from a previous run and
reassigns reads using a statistical model.

#### Basic usage

Basic usage requires a checkpoint file created by an earlier run of
`polymerase assign`. Useful if the run fails after the initial load:

```
polymerase resume [checkpoint]
```

#### Advanced usage

Options are available for tuning the EM optimization, similar to
`polymerase assign`.

```
Input Options:

  checkpoint            Path to checkpoint file.

Reporting Options:

  --quiet               Silence (most) output. (default: False)
  --debug               Print debug messages. (default: False)
  --logfile LOGFILE     Log output to this file. (default: None)
  --outdir OUTDIR       Output directory. (default: .)
  --exp_tag EXP_TAG     Experiment tag (default: polymerase)

Run Modes:

  --reassign_mode {exclude,choose,average,conf,unique}
                        Reassignment mode. After EM is complete, each fragment
                        is reassigned according to the expected value of its
                        membership weights. The reassignment method is the
                        method for resolving the "best" reassignment for
                        fragments that have multiple possible reassignments.
                        Available modes are: "exclude" - fragments with
                        multiple best assignments are excluded from the final
                        counts; "choose" - the best assignment is randomly
                        chosen from among the set of best assignments;
                        "average" - the fragment is divided evenly among the
                        best assignments; "conf" - only assignments that
                        exceed a certain threshold (see --conf_prob) are
                        accepted; "unique" - only uniquely aligned reads are
                        included. NOTE: Results using all assignment modes are
                        included in the Polymerase report by default. This
                        argument determines what mode will be used for the
                        "final counts" column. (default: exclude)
  --use_every_reassign_mode
                        Whether to output count matrices using every reassign mode.
                        If specified, six output count matrices will be generated,
                        corresponding to the six possible reassignment methods (all, exclude,
                        choose, average, conf, unique). (default: False)
  --conf_prob CONF_PROB
                        Minimum probability for high confidence assignment.
                        (default: 0.9)

Model Parameters:

  --pi_prior PI_PRIOR   Prior on pi. Equivalent to adding n unique reads.
                        (default: 0)
  --theta_prior THETA_PRIOR
                        Prior on theta. Equivalent to adding n non-unique reads.
                        (default: 0)
  --em_epsilon EM_EPSILON
                        EM Algorithm Epsilon cutoff (default: 1e-7)
  --max_iter MAX_ITER   EM Algorithm maximum iterations (default: 100)
  --use_likelihood      Use difference in log-likelihood as convergence
                        criteria. (default: False)
```

## Plugin Architecture

Polymerase v2.0.0 introduces a plugin architecture where the core platform
loads BAM files and builds sparse matrices once, then hands that data to any
number of **primers** (analysis plugins) and **cofactors** (post-processors).

### Primers

A primer receives annotation and alignment data via platform hooks, then
produces its own output. The built-in `assign` primer runs the EM algorithm
and writes counts — this is the original Telescope/Polymerase pipeline.

Third-party primers (e.g. chimeric transcript detection, read-through analysis)
can be installed as separate packages and are discovered automatically via
Python entry_points.

### Cofactors

A cofactor transforms a specific primer's committed output. Built-in cofactors:

- **`family-agg`** — aggregates assign counts by `repFamily` and `repClass` from the GTF
- **`normalize`** — computes TPM, RPKM, and CPM from assign counts and feature lengths

Cofactors run automatically after their parent primer commits.

### Writing a Plugin

Primers subclass `polymerase.plugins.Primer` and cofactors subclass
`polymerase.plugins.Cofactor`. Register them in your package's `pyproject.toml`:

```toml
[project.entry-points."polymerase.primers"]
my-primer = "my_package:MyPrimer"

[project.entry-points."polymerase.cofactors"]
my-cofactor = "my_package:MyCofactor"
```

Primers receive frozen snapshots via `on_annotation_loaded()` and
`on_matrix_built()` hooks, then write output in `commit()`. A `ComputeOps`
instance is provided via `configure()` for backend-agnostic math (CPU, Numba,
or GPU automatically).

## Output

Polymerase has three main output files: the transcript counts estimated via EM (`polymerase-TE_counts.tsv`),
a statistical report of the run containing model parameters and additional information
(`polymerase-run_stats.tsv`), and an updated SAM file (optional).
The count file is most important for downstream differential
expression analysis. The updated SAM file is useful for downstream locus-specific analyses.

### Polymerase statistics report

In addition to outputting transcript counts, `polymerase assign`
provides a more detailed
statistical report of each read assignment run.
The first line in the  report is a comment (starting with a "#") that
contains information about the run such as the number of fragments processed,
number of mapped fragments, number of uniquely and ambiguously mapped
fragments, and number of fragments mapping to the annotation. The total number
of mapped fragments may be useful for normalization.

The rest of the report is a table with expression values for
individual transposable element locations calculated using a variety of
reassignment methods, as well as estimated and initial model parameters.
Comparing the results from different assignment methods may shed light on the
model's behaviour. The columns of the table are:

+ `transcript` - Transcript ID, by default from "locus" field. See --attribute argument to use a different attribute.
+ `transcript_length` - Approximate length of transcript. This is calculated from the annotation, not the data, and is equal to the spanning length of the annotation minus any non-model regions.
+ `final_count` - Total number of fragments assigned to transcript after fitting the Polymerase model. This is the column to use for downstream analysis that models data as negative binomial, i.e. DESeq2.
+ `final_conf` - Final confident fragments. The number of fragments assigned to transcript whose posterior probability exceeds a cutoff, 0.9 by default. Set this using the --conf_prob argument.
+ `final_prop` - Final proportion of fragments represented by transcript. This is the final estimate of the pi parameter.
+ `init_aligned` - Initial number of fragments aligned to transcript. A given fragment will contribute +1 to each transcript that it is aligned to, thus the sum of this will be greater than the number of fragments if there are multimapped reads.
+ `unique_count` - Unique count. Number of fragments aligning uniquely to this transcript.
+ `init_best` - Initial number of fragments aligned to transcript that have the "best" alignment score for that fragment. Fragments that have the same best alignment score to multiple transcripts will contribute +1 to each transcript.
+ `init_best_random` - Initial number of fragments aligned to transcript that have the "best" alignment score for that fragment. Fragments that have the same best alignment score to multiple transcripts will be randomly assigned to one transcript.

### Updated SAM file

The updated SAM file contains those fragments that has at least 1 initial
alignment to a transposable element. The final assignment and probabilities are
encoded in the SAM tags:

+ `ZF:Z` Assigned Feature - The name of the feature that alignment is assigned to.
+ `ZT:Z` Polymerase tag - A value of `PRI` indicates that this alignment is the
     best hit for the feature and is used in the likelihood calculations.
     Otherwise the value will be `SEC`, meaning that another alignment to the
     same feature has a higher score.
+ `ZB:Z` Best Feature = The name(s) of the highest scoring feature(s) for the fragment.
+ `YC:Z` Specifies color for alignment as R,G,B.
UCSC sanctioned tag, see documentation
[here.](http://genome.ucsc.edu/goldenpath/help/hgBamTrackHelp.html)
+ `XP:Z` Alignment probability - estimated posterior probability for this alignment.

### Cofactor output

When cofactors are active (the default), additional output files are produced:

**`family-agg` cofactor:**
- `{tag}-family_counts.tsv` — counts aggregated by `repFamily` (e.g. HERVK, L1, Alu)
- `{tag}-class_counts.tsv` — counts aggregated by `repClass` (e.g. LTR, LINE, SINE)

**`normalize` cofactor:**
- `{tag}-normalized_counts.tsv` — per-locus TPM, RPKM, and CPM values alongside raw counts
