# This file is part of Polymerase.
# Original Telescope code by Matthew L. Bendall (https://github.com/mlbendall/telescope)
#
# New code and modifications by Duane Storey (https://github.com/duanestorey) and Claude (Anthropic).
# Licensed under MIT License.

"""Normalization cofactor — computes TPM, RPKM, and CPM from assign counts.

Reads the assign primer's counts and stats TSV files to compute normalized
expression values using feature lengths and total mapped fragment counts.
"""

import logging as lg
import os

import numpy as np
import pandas as pd

from ..abc import Cofactor


class NormalizeCofactor(Cofactor):
    """Compute TPM, RPKM, and CPM from assign counts."""

    @property
    def name(self) -> str:
        return 'normalize'

    @property
    def primer_name(self) -> str:
        return 'assign'

    @property
    def description(self) -> str:
        return 'Compute TPM, RPKM, and CPM normalized counts'

    def configure(self, opts, compute) -> None:
        self._compute = compute

    def transform(self, primer_output_dir, output_dir, exp_tag, console=None, stopwatch=None) -> None:
        counts_file = os.path.join(primer_output_dir, f'{exp_tag}-TE_counts.tsv')
        stats_file = os.path.join(primer_output_dir, f'{exp_tag}-run_stats.tsv')

        if not os.path.exists(counts_file):
            lg.warning(f'normalize: counts file not found at {counts_file}')
            return

        if not os.path.exists(stats_file):
            lg.warning(f'normalize: stats file not found at {stats_file}')
            return

        # Load counts
        counts = pd.read_csv(counts_file, sep='\t')

        # Load stats for feature lengths.
        # The stats file has a RunInfo comment concatenated with the header
        # (no newline between comment and CSV). We parse manually.
        with open(stats_file) as fh:
            raw_lines = fh.readlines()
        # Find the data lines (skip the comment-polluted first line)
        # The first line contains both ## RunInfo... and the column headers
        # merged together. Data lines start from line 2.
        if len(raw_lines) < 2:
            lg.warning('normalize: stats file too short')
            return
        # Extract column names from the first line — they're the last
        # N tab-separated fields that match known column names
        first_fields = raw_lines[0].split('\t')
        # Find where the real columns start
        col_start = 0
        for i, f in enumerate(first_fields):
            # The header "transcript" might be concatenated with the last RunInfo field
            if 'transcript' in f and i > 0:
                # Split off the "transcript" from e.g. "overlap_ambig:1000transcript"
                if f != 'transcript':
                    first_fields[i] = 'transcript'
                col_start = i
                break
        col_names = [f.strip() for f in first_fields[col_start:]]

        # Parse data lines using those columns
        import io

        data_text = '\t'.join(col_names) + '\n' + ''.join(raw_lines[1:])
        stats = pd.read_csv(io.StringIO(data_text), sep='\t')

        # Merge to get lengths
        if 'transcript_length' in stats.columns and 'transcript' in stats.columns:
            lengths = stats[['transcript', 'transcript_length']].drop_duplicates()
            df = counts.merge(lengths, on='transcript', how='left')
        else:
            lg.warning('normalize: stats file missing transcript_length column')
            return

        # Filter out zero-length features and __no_feature
        df = df[df['transcript'] != '__no_feature'].copy()
        df['transcript_length'] = df['transcript_length'].replace(0, np.nan)

        raw_counts = df['count'].values.astype(np.float64)
        lengths_kb = df['transcript_length'].values.astype(np.float64) / 1000.0
        total_mapped = raw_counts.sum()

        # CPM: counts per million mapped fragments
        if total_mapped > 0:
            df['CPM'] = raw_counts / total_mapped * 1e6
        else:
            df['CPM'] = 0.0

        # RPKM: reads per kilobase per million mapped reads
        valid_len = ~np.isnan(lengths_kb) & (lengths_kb > 0)
        df['RPKM'] = 0.0
        if total_mapped > 0:
            df.loc[valid_len, 'RPKM'] = raw_counts[valid_len] / (lengths_kb[valid_len] * total_mapped) * 1e6

        # TPM: transcripts per million
        # rate = count / length, then normalize rates to sum to 1M
        df['TPM'] = 0.0
        rate = np.zeros_like(raw_counts)
        rate[valid_len] = raw_counts[valid_len] / lengths_kb[valid_len]
        rate_sum = rate.sum()
        if rate_sum > 0:
            df['TPM'] = rate / rate_sum * 1e6

        # Round for readability
        for col in ['CPM', 'RPKM', 'TPM']:
            df[col] = df[col].round(4)

        out_name = f'{exp_tag}-normalized_counts.tsv'
        out_file = os.path.join(output_dir, out_name)
        df[['transcript', 'count', 'transcript_length', 'CPM', 'RPKM', 'TPM']].to_csv(out_file, sep='\t', index=False)
        lg.info(f'normalize: wrote {out_file}')

        if console:
            console.detail(f'normalize \u2192 {out_name}')
