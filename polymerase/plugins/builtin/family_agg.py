# This file is part of Polymerase.
# Original Telescope code by Matthew L. Bendall (https://github.com/mlbendall/telescope)
#
# New code and modifications by Duane Storey (https://github.com/duanestorey) and Claude (Anthropic).
# Licensed under MIT License.

"""Family aggregation cofactor — aggregates assign counts by TE family/class.

Reads the assign primer's counts TSV and the original GTF to map each locus
to its ``repFamily`` and ``repClass`` attributes, then sums counts.
"""

import logging as lg
import os

import pandas as pd

from ...annotation.gtf_utils import detect_family_class_attrs
from ..abc import Cofactor


def _parse_gtf_attributes(attr_str):
    """Parse GTF attribute string into a dict."""
    attrs = {}
    for field in attr_str.strip().rstrip(';').split(';'):
        field = field.strip()
        if not field:
            continue
        parts = field.split(' ', 1)
        if len(parts) == 2:
            key = parts[0]
            val = parts[1].strip('"')
            attrs[key] = val
    return attrs


def _build_locus_metadata(gtf_path, locus_attr='locus'):
    """Build mapping from locus name to family/class from GTF exon lines.

    Auto-detects whether the GTF uses repFamily/repClass or family_id/class_id.

    Returns:
        dict: {locus_name: {'repFamily': str, 'repClass': str}}
    """
    family_key, class_key = detect_family_class_attrs(gtf_path)
    lg.debug(f'family-agg: detected GTF attributes: family={family_key}, class={class_key}')

    meta = {}
    with open(gtf_path) as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue
            if parts[2] != 'exon':
                continue
            attrs = _parse_gtf_attributes(parts[8])
            locus = attrs.get(locus_attr)
            if locus and locus not in meta:
                meta[locus] = {
                    'repFamily': attrs.get(family_key, 'Unknown'),
                    'repClass': attrs.get(class_key, 'Unknown'),
                }
    return meta


class FamilyAggCofactor(Cofactor):
    """Aggregate assign counts by TE family and class."""

    @property
    def name(self) -> str:
        return 'family-agg'

    @property
    def primer_name(self) -> str:
        return 'assign'

    @property
    def description(self) -> str:
        return 'Aggregate TE counts by repFamily and repClass'

    def configure(self, opts, compute) -> None:
        self._gtf_path = getattr(opts, 'gtffile', None)
        self._attribute = getattr(opts, 'attribute', 'locus')

    def transform(self, primer_output_dir, output_dir, exp_tag, console=None, stopwatch=None) -> None:
        # Find assign counts file
        counts_file = os.path.join(primer_output_dir, f'{exp_tag}-TE_counts.tsv')
        if not os.path.exists(counts_file):
            lg.warning(f'family-agg: counts file not found at {counts_file}')
            return

        if not self._gtf_path or not os.path.exists(self._gtf_path):
            lg.warning('family-agg: GTF file not available, skipping')
            return

        # Load counts
        counts = pd.read_csv(counts_file, sep='\t')
        if 'transcript' not in counts.columns or 'count' not in counts.columns:
            lg.warning('family-agg: unexpected counts format')
            return

        # Build locus -> family/class mapping
        meta = _build_locus_metadata(self._gtf_path, self._attribute)

        # Add family/class columns
        counts['repFamily'] = counts['transcript'].map(lambda t: meta.get(t, {}).get('repFamily', 'Unknown'))
        counts['repClass'] = counts['transcript'].map(lambda t: meta.get(t, {}).get('repClass', 'Unknown'))

        # Aggregate by family
        family_counts = counts.groupby('repFamily')['count'].sum().reset_index()
        family_counts.columns = ['repFamily', 'count']
        family_counts.sort_values('repFamily', inplace=True)

        family_name = f'{exp_tag}-family_counts.tsv'
        family_file = os.path.join(output_dir, family_name)
        family_counts.to_csv(family_file, sep='\t', index=False)
        lg.info(f'family-agg: wrote {family_file}')

        # Aggregate by class
        class_counts = counts.groupby('repClass')['count'].sum().reset_index()
        class_counts.columns = ['repClass', 'count']
        class_counts.sort_values('repClass', inplace=True)

        class_name = f'{exp_tag}-class_counts.tsv'
        class_file = os.path.join(output_dir, class_name)
        class_counts.to_csv(class_file, sep='\t', index=False)
        lg.info(f'family-agg: wrote {class_file}')

        if console:
            console.detail(f'family-agg \u2192 {family_name}, {class_name}')
