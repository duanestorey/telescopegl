# This file is part of Polymerase.
# Original Telescope code by Matthew L. Bendall (https://github.com/mlbendall/telescope)
#
# New code and modifications by Duane Storey (https://github.com/duanestorey) and Claude (Anthropic).
# Licensed under MIT License.

"""Shared GTF utilities for attribute auto-detection."""

import re


def detect_family_class_attrs(gtf_path, feature_types=None):
    """Probe the first matching line in a GTF to detect family/class attribute names.

    Supports two naming conventions:
    - retro.hg38 format: ``repFamily`` / ``repClass``
    - RepeatMasker standard format: ``family_id`` / ``class_id``

    Args:
        gtf_path: Path to GTF file.
        feature_types: Set of GTF feature types to consider. Defaults to ``{'exon'}``.

    Returns:
        (family_key, class_key) tuple, or ('repFamily', 'repClass') as default.
    """
    if feature_types is None:
        feature_types = {'exon'}
    with open(gtf_path) as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue
            if parts[2] not in feature_types:
                continue
            attrs = dict(re.findall(r'(\w+)\s+"(.+?)";', parts[8]))
            if 'family_id' in attrs:
                return ('family_id', 'class_id')
            return ('repFamily', 'repClass')
    # Fallback if no matching lines found
    return ('repFamily', 'repClass')
