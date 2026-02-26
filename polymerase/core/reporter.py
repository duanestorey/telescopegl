# This file is part of Polymerase.
# Original Telescope code by Matthew L. Bendall (https://github.com/mlbendall/telescope)
#
# New code and modifications by Duane Storey (https://github.com/duanestorey) and Claude (Anthropic).
# Licensed under MIT License.

"""Report generation and SAM file updating for Polymerase.

Functions accept individual data pieces rather than full Polymerase objects,
so they can be called from the assign primer without coupling.
"""

import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pysam

from ..alignment import fragments as alignment
from ..sparse.backend import get_backend
from ..sparse.matrix import csr_matrix_plus as csr_matrix
from ..utils.colors import D2PAL, GPAL, c2str
from ..utils.helpers import phred


def output_report(feat_index, feature_length, run_info, tl, reassign_mode, conf_prob, stats_filename, counts_filename):
    """Generate TSV stats and counts reports.

    Args:
        feat_index: dict {feature_name: col_index}
        feature_length: dict {feature_name: length_bp}
        run_info: OrderedDict of alignment statistics
        tl: PolymeraseLikelihood instance (after EM convergence)
        reassign_mode: str reassignment method
        conf_prob: float confidence threshold
        stats_filename: Path for stats TSV output
        counts_filename: Path for counts TSV output
    """
    _rmethod = reassign_mode
    _rprob = conf_prob
    _fnames = sorted(feat_index, key=feat_index.get)
    _flens = feature_length
    _stats_rounding = pd.Series([2, 3, 2, 3], index=['final_conf', 'final_prop', 'init_best_avg', 'init_prop'])

    # Define reassignment tasks: (key, method, thresh, initial)
    _tasks = [
        ('final_conf', 'conf', _rprob, False),
        ('init_aligned', 'all', 0.9, True),
        ('unique_count', 'unique', 0.9, False),
        ('init_best', 'exclude', 0.9, True),
        ('init_best_random', 'choose', 0.9, True),
        ('init_best_avg', 'average', 0.9, True),
    ]

    def _do_reassign(task):
        key, method, thresh, initial = task
        return key, tl.reassign(method, thresh, initial).sum(0).A1

    backend = get_backend()
    if backend.threadsafe_parallelism:
        with ThreadPoolExecutor(max_workers=len(_tasks)) as pool:
            reassign_results = dict(pool.map(_do_reassign, _tasks))
    else:
        reassign_results = dict(map(_do_reassign, _tasks))

    _stats_report0 = {
        'transcript': _fnames,
        'transcript_length': [_flens.get(f, 0) for f in _fnames],
        'final_conf': reassign_results['final_conf'],
        'final_prop': tl.pi,
        'init_aligned': reassign_results['init_aligned'],
        'unique_count': reassign_results['unique_count'],
        'init_best': reassign_results['init_best'],
        'init_best_random': reassign_results['init_best_random'],
        'init_best_avg': reassign_results['init_best_avg'],
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

    _comment = ['## RunInfo']
    _comment += ['{}:{}'.format(*tup) for tup in run_info.items()]

    with open(stats_filename, 'w') as outh:
        outh.write('\t'.join(_comment))
        _stats_report.to_csv(outh, sep='\t', index=False)

    with open(counts_filename, 'w') as outh:
        _counts.to_csv(outh, sep='\t', index=False)


def update_sam(tmp_bam, read_index, feat_index, run_info, opts, tl, filename):
    """Update SAM file with EM-derived assignments and quality scores.

    Args:
        tmp_bam: Path to temporary BAM with overlapping fragments
        read_index: dict {fragment_name: row_index}
        feat_index: dict {feature_name: col_index}
        run_info: OrderedDict of alignment statistics
        opts: Options object with reassign_mode, conf_prob, ncpu
        tl: PolymeraseLikelihood instance (after EM convergence)
        filename: Path for output BAM file
    """
    _rmethod = opts.reassign_mode
    _rprob = opts.conf_prob

    mat = csr_matrix(tl.reassign(_rmethod, _rprob))

    _threads = getattr(opts, 'ncpu', 1)
    with pysam.AlignmentFile(tmp_bam, check_sq=False, threads=_threads) as sf:
        header = sf.header
        header['PG'].append(
            {
                'PN': 'polymerase',
                'ID': 'polymerase',
                'VN': run_info['version'],
                'CL': ' '.join(sys.argv),
            }
        )
        outsam = pysam.AlignmentFile(filename, 'wb', header=header)
        for _code, pairs in alignment.fetch_fragments_seq(sf, until_eof=True):
            if len(pairs) == 0:
                continue
            ridx = read_index[pairs[0].query_id]
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
                    fidx = feat_index[aln.r1.get_tag('ZF')]
                    prob = tl.z[ridx, fidx]
                    aln.set_mapq(phred(prob))
                    aln.set_tag('XP', int(round(prob * 100)))
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
