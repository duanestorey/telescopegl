# -*- coding: utf-8 -*-

# This file is part of Polymerase.
# Original Telescope code by Matthew L. Bendall (https://github.com/mlbendall/telescope)
#
# New code and modifications by Duane Storey (https://github.com/duanestorey) and Claude (Anthropic).
# Licensed under MIT License.

"""Frozen dataclass snapshots passed to primers via platform hooks."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AnnotationSnapshot:
    """Immutable snapshot of parsed annotation data.

    Passed to :meth:`Primer.on_annotation_loaded`.
    """
    loci: dict                    # {locus_name: [GTFRow, ...]}
    feature_lengths: dict         # {locus_name: int}
    num_features: int
    gtf_path: str
    attribute_name: str


@dataclass(frozen=True)
class AlignmentSnapshot:
    """Immutable snapshot of alignment / matrix data.

    Passed to :meth:`Primer.on_matrix_built`.
    """
    bam_path: str                 # primers needing BAM open their own handle
    run_info: dict                # alignment stats (total reads, overlaps, etc.)
    read_index: dict              # {fragment_name: row_index}
    feat_index: dict              # {feature_name: col_index}
    shape: tuple                  # (N_fragments, K_features)
    raw_scores: object            # CSR sparse matrix (read-only by convention)
    feature_lengths: dict         # {feature_name: length_bp}
