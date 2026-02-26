# -*- coding: utf-8 -*-
"""Report generation and SAM file updating for Telescope.

Extracted from model.py to separate I/O concerns from core algorithm.
"""
import sys

import pandas as pd
import pysam

from .sparse_plus import csr_matrix_plus as csr_matrix
from .colors import c2str, D2PAL, GPAL
from .helpers import phred
from . import alignment

__author__ = 'Matthew L. Bendall'
__copyright__ = "Copyright (C) 2019 Matthew L. Bendall"


def output_report(telescope_obj, tl, stats_filename, counts_filename):
    """Generate TSV stats and counts reports.

    Args:
        telescope_obj: Telescope instance with feat_index, feature_length, run_info, opts
        tl: TelescopeLikelihood instance (after EM convergence)
        stats_filename: Path for stats TSV output
        counts_filename: Path for counts TSV output
    """
    _rmethod = telescope_obj.opts.reassign_mode
    _rprob = telescope_obj.opts.conf_prob
    _fnames = sorted(telescope_obj.feat_index, key=telescope_obj.feat_index.get)
    _flens = telescope_obj.feature_length
    _stats_rounding = pd.Series(
        [2, 3, 2, 3],
        index=['final_conf', 'final_prop', 'init_best_avg', 'init_prop']
    )

    _stats_report0 = {
        'transcript': _fnames,
        'transcript_length': [_flens[f] for f in _fnames],
        'final_conf': tl.reassign('conf', _rprob).sum(0).A1,
        'final_prop': tl.pi,
        'init_aligned': tl.reassign('all', initial=True).sum(0).A1,
        'unique_count': tl.reassign('unique').sum(0).A1,
        'init_best': tl.reassign('exclude', initial=True).sum(0).A1,
        'init_best_random': tl.reassign('choose', initial=True).sum(0).A1,
        'init_best_avg': tl.reassign('average', initial=True).sum(0).A1,
        'init_prop': tl.pi_init,
    }

    _stats_report = pd.DataFrame(_stats_report0)
    _stats_report.sort_values('final_prop', ascending=False, inplace=True)
    _stats_report = _stats_report.round(_stats_rounding)

    _counts0 = {
        'transcript': _fnames,
        'count': tl.reassign(_rmethod, _rprob).sum(0).A1,
    }
    _counts = pd.DataFrame(_counts0)
    _counts.sort_values('transcript', inplace=True)

    _comment = ["## RunInfo"]
    _comment += ['{}:{}'.format(*tup) for tup in telescope_obj.run_info.items()]

    with open(stats_filename, 'w') as outh:
        outh.write('\t'.join(_comment))
        _stats_report.to_csv(outh, sep='\t', index=False)

    with open(counts_filename, 'w') as outh:
        _counts.to_csv(outh, sep='\t', index=False)


def update_sam(telescope_obj, tl, filename):
    """Update SAM file with EM-derived assignments and quality scores.

    Args:
        telescope_obj: Telescope instance with tmp_bam, read_index, feat_index, run_info, opts
        tl: TelescopeLikelihood instance (after EM convergence)
        filename: Path for output BAM file
    """
    _rmethod = telescope_obj.opts.reassign_mode
    _rprob = telescope_obj.opts.conf_prob
    _fnames = sorted(telescope_obj.feat_index, key=telescope_obj.feat_index.get)

    mat = csr_matrix(tl.reassign(_rmethod, _rprob))

    with pysam.AlignmentFile(telescope_obj.tmp_bam, check_sq=False) as sf:
        header = sf.header
        header['PG'].append({
            'PN': 'telescope', 'ID': 'telescope',
            'VN': telescope_obj.run_info['version'],
            'CL': ' '.join(sys.argv),
        })
        outsam = pysam.AlignmentFile(filename, 'wb', header=header)
        for code, pairs in alignment.fetch_fragments_seq(sf, until_eof=True):
            if len(pairs) == 0:
                continue
            ridx = telescope_obj.read_index[pairs[0].query_id]
            for aln in pairs:
                if aln.is_unmapped:
                    aln.write(outsam)
                    continue
                assert aln.r1.has_tag('ZT'), 'Missing ZT tag'
                if aln.r1.get_tag('ZT') == 'SEC':
                    aln.set_flag(pysam.FSECONDARY)
                    aln.set_tag('YC', c2str((248, 248, 248)))
                    aln.set_mapq(0)
                else:
                    fidx = telescope_obj.feat_index[aln.r1.get_tag('ZF')]
                    prob = tl.z[ridx, fidx]
                    aln.set_mapq(phred(prob))
                    aln.set_tag('XP', int(round(prob*100)))
                    if mat[ridx, fidx] > 0:
                        aln.unset_flag(pysam.FSECONDARY)
                        aln.set_tag('YC', c2str(D2PAL['vermilion']))
                    else:
                        aln.set_flag(pysam.FSECONDARY)
                        if prob >= 0.2:
                            aln.set_tag('YC', c2str(D2PAL['yellow']))
                        else:
                            aln.set_tag('YC', c2str(GPAL[2]))
                aln.write(outsam)
        outsam.close()
