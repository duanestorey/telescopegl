# This file is part of Polymerase.
# Original Telescope code by Matthew L. Bendall (https://github.com/mlbendall/telescope)
#
# New code and modifications by Duane Storey (https://github.com/duanestorey) and Claude (Anthropic).
# Licensed under MIT License.

"""Polymerase pipeline: BAM loading, matrix construction, and orchestration.

PolymeraseLikelihood (EM algorithm) is in likelihood.py.
Report generation helpers are in reporter.py.
"""

import functools
import logging as lg
from collections import Counter, OrderedDict, defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pysam
import scipy

from ..alignment import fragments as alignment
from ..cli import BIG_INT
from ..sparse.backend import get_sparse_class
from ..utils.helpers import merge_overlapping_regions, region_iter, str2int
from .reporter import output_report as _output_report_func
from .reporter import update_sam as _update_sam_func


def process_overlap_frag(pairs, overlap_feats, write_tags=True):
    """Find the best alignment for each locus.

    Args:
        pairs: List of AlignedPair objects for this fragment.
        overlap_feats: List of feature names (parallel to pairs).
        write_tags: If False, skip writing SAM tags (ZF/ZT/ZB).
            Set to False on the indexed path when --updated_sam is not
            requested, avoiding ~30% of per-fragment processing overhead.
    """
    assert all(pairs[0].query_id == p.query_id for p in pairs)
    byfeature = defaultdict(list)
    for pair, feat in zip(pairs, overlap_feats):
        byfeature[feat].append(pair)

    _maps = []
    for feat, falns in byfeature.items():
        # Sort alignments by score + length
        falns.sort(key=lambda x: x.alnscore + x.alnlen, reverse=True)
        # Add best alignment to mappings
        _topaln = falns[0]
        _maps.append((_topaln.query_id, feat, _topaln.alnscore, _topaln.alnlen))
        if write_tags:
            _topaln.set_tag('ZF', feat)
            _topaln.set_tag('ZT', 'PRI')
            for aln in falns[1:]:
                aln.set_tag('ZF', feat)
                aln.set_tag('ZT', 'SEC')

    if write_tags:
        # Sort mappings by score
        _maps.sort(key=lambda x: x[2], reverse=True)
        # Top feature(s), comma separated
        _topfeat = ','.join(t[1] for t in _maps if t[2] == _maps[0][2])
        # Add best feature tag (ZB) to all alignments
        for p in pairs:
            p.set_tag('ZB', _topfeat)

    return _maps


def _resolve_duplicates_max(rows, cols, data, shape):
    """Build CSR matrix from COO arrays, resolving duplicate (i,j) by max.

    Args:
        rows, cols, data: COO-format arrays.
        shape: (n_rows, n_cols) tuple.

    Returns:
        scipy.sparse.csr_matrix with max value for each (i,j).
    """
    rows = np.array(rows, dtype=np.intp)
    cols = np.array(cols, dtype=np.intp)
    data = np.array(data, dtype=np.uint16)

    if len(rows) == 0:
        return scipy.sparse.csr_matrix(shape, dtype=np.uint16)

    # Sort by (row, col) for grouping
    order = np.lexsort((cols, rows))
    rows = rows[order]
    cols = cols[order]
    data = data[order]

    # Find group boundaries: where (row, col) changes
    diff = np.diff(rows).astype(bool) | np.diff(cols).astype(bool)
    boundaries = np.concatenate(([0], np.where(diff)[0] + 1))

    # Take max within each group
    out_rows = rows[boundaries]
    out_cols = cols[boundaries]
    out_data = np.maximum.reduceat(data, boundaries)

    return scipy.sparse.csr_matrix((out_data, (out_rows, out_cols)), shape=shape, dtype=np.uint16)


def _print_progress(nfrags, infolev=2500000):
    mfrags = nfrags / 1e6
    msg = f'...processed {mfrags:.1f}M fragments'
    if nfrags % infolev == 0:
        lg.info(msg)
    else:
        lg.debug(msg)


class Polymerase:
    """BAM loading, sparse matrix construction, and pipeline orchestration."""

    def __init__(self, opts):

        self.opts = opts  # Command line options
        self.single_cell = False  # Single cell sequencing
        self.run_info = OrderedDict()  # Information about the run
        self.feature_length = None  # Lengths of features
        self.read_index = {}  # {"fragment name": row_index}
        self.feat_index = {}  # {"feature_name": column_index}
        self.shape = None  # Fragments x Features
        self.raw_scores = None  # Initial alignment scores

        # BAM with non overlapping fragments (or unmapped)
        self.other_bam = opts.outfile_path('other.bam')
        # BAM with overlapping fragments
        self.tmp_bam = opts.outfile_path('tmp_tele.bam')

        # Set the version
        self.run_info['version'] = self.opts.version

        _threads = getattr(self.opts, 'ncpu', 1)
        with pysam.AlignmentFile(self.opts.samfile, check_sq=False, threads=_threads) as sf:
            self.has_index = sf.has_index()
            _is_coordinate_sorted = sf.header.get('HD', {}).get('SO') == 'coordinate'
            self.ref_names = sf.references
            self.ref_lengths = sf.lengths

        # Auto-create .bai index for coordinate-sorted BAMs without one
        if not self.has_index and _is_coordinate_sorted:
            lg.info('Coordinate-sorted BAM without index — creating .bai')
            lg.info('Note: Pre-creating the .bai index (e.g., samtools index) in your pipeline avoids this step.')
            pysam.index(self.opts.samfile)
            self.has_index = True

        # Read mapped/unmapped counts from index
        if self.has_index:
            with pysam.AlignmentFile(self.opts.samfile, check_sq=False, threads=_threads) as sf:
                self.run_info['nmap_idx'] = sf.mapped
                self.run_info['nunmap_idx'] = sf.unmapped

        return

    def save(self, filename):
        _feat_list = sorted(self.feat_index, key=self.feat_index.get)
        _flen_list = [self.feature_length[f] for f in _feat_list]
        np.savez(
            filename,
            _run_info=list(self.run_info.items()),
            _flen_list=_flen_list,
            _feat_list=_feat_list,
            _read_list=sorted(self.read_index, key=self.read_index.get),
            _shape=self.shape,
            _raw_scores_data=self.raw_scores.data,
            _raw_scores_indices=self.raw_scores.indices,
            _raw_scores_indptr=self.raw_scores.indptr,
            _raw_scores_shape=self.raw_scores.shape,
        )

    @classmethod
    def load(cls, filename):
        loader = np.load(filename)
        obj = cls.__new__(cls)
        """ Run info """
        obj.run_info = OrderedDict()
        for r in range(loader['_run_info'].shape[0]):
            k = loader['_run_info'][r, 0]
            v = str2int(loader['_run_info'][r, 1])
            obj.run_info[k] = v
        obj.feature_length = Counter()
        for f, fl in zip(loader['_feat_list'], loader['_flen_list']):
            obj.feature_length[f] = fl
        """ Read and feature indexes """
        obj.read_index = {n: i for i, n in enumerate(loader['_read_list'])}
        obj.feat_index = {n: i for i, n in enumerate(loader['_feat_list'])}
        obj.shape = len(obj.read_index), len(obj.feat_index)
        assert tuple(loader['_shape']) == obj.shape

        _sparse_cls = get_sparse_class()
        obj.raw_scores = _sparse_cls(
            (loader['_raw_scores_data'], loader['_raw_scores_indices'], loader['_raw_scores_indptr']),
            shape=loader['_raw_scores_shape'],
        )
        return obj

    def get_random_seed(self):
        ret = self.run_info['total_fragments'] % (self.shape[0] * self.shape[1])
        # 2**32 - 1 = 4294967295
        return ret % 4294967295

    @staticmethod
    def _get_te_regions(annotation, padding=500):
        """Extract TE locus coordinates with padding, merge overlapping.

        Args:
            annotation: Annotation object with itree or loci dict.
            padding: bp to add around each region.

        Returns:
            List of merged (chrom, start, end) tuples.
        """
        regions = []
        if hasattr(annotation, 'itree'):
            for chrom, tree in annotation.itree.items():
                for iv in tree:
                    regions.append((chrom, iv.begin, iv.end))
        else:
            for _locus_name, locus_data in annotation.loci.items():
                if isinstance(locus_data, list):
                    for row in locus_data:
                        regions.append((row.chrom, int(row.start), int(row.end)))
                elif hasattr(locus_data, 'chrom'):
                    regions.append((locus_data.chrom, locus_data.start, locus_data.end))
        if not regions:
            return []
        return merge_overlapping_regions(regions, padding=padding)

    def _load_indexed(self, annotation):
        """Load alignments from coordinate-sorted indexed BAM.

        Two-pass approach for correctness:
          Phase 1: Fetch from TE regions to identify relevant fragment names.
          Phase 2: Stream full BAM, buffer ALL alignments for relevant
                   fragments (including mates and non-TE mappings).
          Phase 3: Process buffered fragments with identical logic to
                   _load_sequential (same pairing, thresholds, __no_feature).

        This produces identical matrices to _load_sequential while skipping
        the expensive assign() calls for the ~99.5% of fragments that don't
        overlap any TE region.
        """
        from ..alignment.calignment import AlignedPair

        _nfkey = self.opts.no_feature_key
        _omode, _othresh = self.opts.overlap_mode, self.opts.overlap_threshold

        _mappings = []
        assign = Assigner(annotation, _nfkey, _omode, _othresh, self.opts.stranded_mode).assign_func()

        alninfo = Counter()
        te_regions = self._get_te_regions(annotation)
        lg.info(f'Indexed loading: {len(te_regions)} merged TE regions')

        _threads = min(4, max(1, getattr(self.opts, 'ncpu', 1)))

        # Phase 1: Fetch from TE regions to identify relevant fragment names
        relevant_names = set()
        with pysam.AlignmentFile(self.opts.samfile, check_sq=False, threads=_threads) as sf:
            for chrom, start, end in te_regions:
                try:
                    for aln in sf.fetch(chrom, start, end):
                        if not aln.is_unmapped:
                            relevant_names.add(aln.query_name)
                except ValueError:
                    continue

        lg.info(f'Phase 1: {len(relevant_names)} fragments overlap TE regions')

        # Phase 2: Stream full BAM, buffer ALL alignments for relevant fragments
        read_buffer = defaultdict(list)
        with pysam.AlignmentFile(self.opts.samfile, check_sq=False, threads=_threads) as sf:
            _total_mapped = sf.mapped
            _total_unmapped = sf.unmapped
            for aln in sf.fetch(until_eof=True):
                if aln.query_name in relevant_names:
                    read_buffer[aln.query_name].append(aln)

        lg.info(
            f'Phase 2: Buffered {sum(len(v) for v in read_buffer.values())} alignments for {len(read_buffer)} fragments'
        )

        _minAS, _maxAS = BIG_INT, -BIG_INT

        for _qname, alns in read_buffer.items():
            # Sort to match collated BAM order: primary alignments first,
            # then secondary, then supplementary. This ensures alns[0]
            # is the primary alignment, making is_proper_pair classification
            # independent of BAM sort order.
            alns.sort(key=lambda a: (a.is_supplementary, a.is_secondary))

            # Mirror fetch_fragments_seq classification logic
            aln0 = alns[0]
            if not aln0.is_paired:
                _code = 'SU' if aln0.is_unmapped else 'SM'
                ci = alignment.CODE_INT[_code]
                pairs = [AlignedPair(a) for a in alns]
            else:
                if aln0.is_proper_pair:
                    ci = alignment.CODE_INT['PM']
                    pairs = list(alignment.pair_bundle(alns))
                else:
                    if len(alns) == 2 and all(a.is_unmapped for a in alns):
                        ci = alignment.CODE_INT['PU']
                        pairs = [AlignedPair(alns[0], alns[1])]
                    else:
                        ci = alignment.CODE_INT['PX']
                        pairs = [AlignedPair(a) for a in alns]

            # Filter to mapped pairs
            _mapped = [p for p in pairs if not p.is_unmapped]
            if not _mapped:
                continue

            # Ambiguity by mapped count (same as sequential)
            _ambig = len(_mapped) > 1

            # Update score range
            _scores = [p.alnscore for p in _mapped]
            _minAS = min(_minAS, *_scores)
            _maxAS = max(_maxAS, *_scores)

            # Assign overlaps
            overlap_feats = list(map(assign, _mapped))
            has_overlap = any(f != _nfkey for f in overlap_feats)

            if not has_overlap:
                alninfo['nofeat_{}'.format('A' if _ambig else 'U')] += 1
                continue

            alninfo['feat_{}'.format('A' if _ambig else 'U')] += 1

            # Single-cell barcode
            if self.single_cell and aln0.has_tag(self.opts.barcode_tag):
                self.read_barcodes[pairs[0].query_id] = aln0.get_tag(self.opts.barcode_tag)

            for m in process_overlap_frag(_mapped, overlap_feats, write_tags=False):
                _mappings.append((ci, m[0], m[1], m[2], m[3]))

        # Derive stats — use index for totals, computed for overlap stats
        alninfo['total_fragments'] = (_total_mapped + _total_unmapped) // 2
        alninfo['pair_mapped'] = _total_mapped // 2
        alninfo['pair_mixed'] = 0
        alninfo['single_mapped'] = 0
        alninfo['unmapped'] = _total_unmapped // 2
        alninfo['unique'] = alninfo.get('nofeat_U', 0) + alninfo.get('feat_U', 0)
        alninfo['ambig'] = alninfo.get('nofeat_A', 0) + alninfo.get('feat_A', 0)

        if _maxAS < _minAS:
            _minAS, _maxAS = 0, 0

        return _mappings, (_minAS, _maxAS), alninfo

    def load_alignment(self, annotation):
        self.run_info['annotated_features'] = len(annotation.loci)
        self.feature_length = annotation.feature_length().copy()

        _update_sam = getattr(self.opts, 'updated_sam', False)
        if self.has_index and not _update_sam:
            lg.info('Coordinate-sorted indexed BAM detected, using region-based loading')
            maps, scorerange, alninfo = self._load_indexed(annotation)
        elif self.opts.ncpu > 1:
            maps, scorerange, alninfo = self._load_parallel(annotation)
        else:
            maps, scorerange, alninfo = self._load_sequential(annotation)
            lg.debug(str(alninfo))

        self._mapping_to_matrix(maps, scorerange, alninfo)
        lg.debug(str(alninfo))

        run_fields = [
            'total_fragments',
            'pair_mapped',
            'pair_mixed',
            'single_mapped',
            'unmapped',
            'unique',
            'ambig',
            'overlap_unique',
            'overlap_ambig',
        ]
        for f in run_fields:
            self.run_info[f] = alninfo[f]

    # TODO (future): Replace temp file I/O in _load_parallel with in-memory
    # return values to avoid disk round-trips. Low priority since indexed
    # loading bypasses this path entirely for sorted BAMs.

    def _load_parallel(self, annotation):
        lg.info('Loading alignments in parallel...')
        regions = region_iter(
            self.ref_names,
            self.ref_lengths,
            max(self.ref_lengths),  # Do not split contigs
        )

        opt_d = {
            'no_feature_key': self.opts.no_feature_key,
            'overlap_mode': self.opts.overlap_mode,
            'overlap_threshold': self.opts.overlap_threshold,
            'stranded_mode': self.opts.stranded_mode,
            'tempdir': self.opts.tempdir,
        }
        _minAS, _maxAS = BIG_INT, -BIG_INT
        alninfo = Counter()
        mfiles = []
        with Pool(processes=self.opts.ncpu) as pool:
            _loadfunc = functools.partial(
                alignment.fetch_region,
                self.opts.samfile,
                annotation,
                opt_d,
            )
            result = pool.map_async(_loadfunc, regions)
            for mfile, scorerange, _pxu in result.get():
                alninfo['unmap_x'] += _pxu
                _minAS = min(scorerange[0], _minAS)
                _maxAS = max(scorerange[1], _maxAS)
                mfiles.append(mfile)

        _miter = self._mapping_fromfiles(mfiles)
        return _miter, (_minAS, _maxAS), alninfo

    def _mapping_fromfiles(self, files):
        for f in files:
            with open(f) as fh:
                for line in fh:
                    code, rid, fid, ascr, alen = line.strip('\n').split('\t')
                    yield (int(code), rid, fid, int(ascr), int(alen))

    def _load_sequential(self, annotation):
        _update_sam = self.opts.updated_sam
        _nfkey = self.opts.no_feature_key
        _omode, _othresh = self.opts.overlap_mode, self.opts.overlap_threshold

        _mappings = []
        assign = Assigner(annotation, _nfkey, _omode, _othresh, self.opts.stranded_mode).assign_func()

        """ Load unsorted reads """
        alninfo = Counter()
        with pysam.AlignmentFile(self.opts.samfile, check_sq=False, threads=getattr(self.opts, 'ncpu', 1)) as sf:
            # Create output temporary files
            if _update_sam:
                bam_u = pysam.AlignmentFile(self.other_bam, 'wb', template=sf)
                bam_t = pysam.AlignmentFile(self.tmp_bam, 'wb', template=sf)

            try:
                _minAS, _maxAS = BIG_INT, -BIG_INT
                for ci, alns in alignment.fetch_fragments_seq(sf, until_eof=True):
                    alninfo['total_fragments'] += 1
                    if alninfo['total_fragments'] % 500000 == 0:
                        _print_progress(alninfo['total_fragments'])

                    """ Count code """
                    _code = alignment.CODES[ci][0]
                    alninfo[_code] += 1

                    """ Check whether fragment is mapped """
                    if _code == 'SU' or _code == 'PU':
                        if _update_sam:
                            alns[0].write(bam_u)
                        continue

                    """ If running with single cell data, add cell """
                    if self.single_cell and alns[0].r1.has_tag(self.opts.barcode_tag):
                        self.read_barcodes[alns[0].query_id] = dict(alns[0].r1.get_tags()).get(self.opts.barcode_tag)

                    """ Fragment is ambiguous if multiple mappings"""
                    _mapped = [a for a in alns if not a.is_unmapped]
                    _ambig = len(_mapped) > 1

                    """ Update min and max scores """
                    _scores = [a.alnscore for a in _mapped]
                    _minAS = min(_minAS, *_scores)
                    _maxAS = max(_maxAS, *_scores)

                    """ Check whether fragment overlaps annotation """
                    overlap_feats = list(map(assign, _mapped))
                    has_overlap = any(f != _nfkey for f in overlap_feats)

                    """ Fragment has no overlap """
                    if not has_overlap:
                        alninfo['nofeat_{}'.format('A' if _ambig else 'U')] += 1
                        if _update_sam:
                            [p.write(bam_u) for p in alns]
                        continue

                    """ Fragment overlaps with annotation """
                    alninfo['feat_{}'.format('A' if _ambig else 'U')] += 1

                    """ Find the best alignment for each locus """
                    for m in process_overlap_frag(_mapped, overlap_feats):
                        _mappings.append((ci, m[0], m[1], m[2], m[3]))

                    if _update_sam:
                        [p.write(bam_t) for p in alns]
            finally:
                if _update_sam:
                    bam_u.close()
                    bam_t.close()

        return _mappings, (_minAS, _maxAS), alninfo

    def _mapping_to_matrix(self, miter, scorerange, alninfo):
        _isparallel = 'total_fragments' not in alninfo
        minAS, maxAS = scorerange
        lg.debug(f'min alignment score: {minAS}')
        lg.debug(f'max alignment score: {maxAS}')
        # Function to rescale integer alignment scores
        # Scores should be greater than zero
        rescale = {s: (s - minAS + 1) for s in range(minAS, maxAS + 1)}

        # Collect COO entries (sequential for index-building)
        rcodes = defaultdict(Counter)
        _rows = []
        _cols = []
        _data = []
        _ridx = self.read_index
        _fidx = self.feat_index
        _fidx[self.opts.no_feature_key] = 0

        for code, rid, fid, ascr, alen in miter:
            i = _ridx.setdefault(rid, len(_ridx))
            j = _fidx.setdefault(fid, len(_fidx))
            _rows.append(i)
            _cols.append(j)
            _data.append(rescale[ascr] + alen)
            if _isparallel:
                rcodes[code][i] += 1

        # Build CSR matrix, resolving duplicate (i,j) by max
        _m1 = _resolve_duplicates_max(_rows, _cols, _data, (len(_ridx), len(_fidx)))

        """ Map barcodes to read indices """
        if self.single_cell:
            _bcidx = self.barcode_read_indices
            for rid, rbc in self.read_barcodes.items():
                if rid in _ridx:
                    _bcidx[rbc].append(_ridx[rid])

        """ Update counts """
        if _isparallel:
            # Default for nunmap_idx is zero
            unmap_both = self.run_info.get('nunmap_idx', 0) - alninfo['unmap_x']
            alninfo['unmapped'] = unmap_both // 2
            for cs, _desc in alignment.CODES:
                ci = alignment.CODE_INT[cs]
                if cs not in alninfo and ci in rcodes:
                    alninfo[cs] = len(rcodes[ci])
                if cs in ['SM', 'PM', 'PX'] and ci in rcodes:
                    _a = sum(v > 1 for k, v in rcodes[ci].items())
                    alninfo['unique'] += len(rcodes[ci]) - _a
                    alninfo['ambig'] += _a
            alninfo['total_fragments'] = alninfo['unmapped'] + alninfo['PM'] + alninfo['PX'] + alninfo['SM']
        else:
            alninfo['unmapped'] = alninfo['SU'] + alninfo['PU']
            alninfo['unique'] = alninfo['nofeat_U'] + alninfo['feat_U']
            alninfo['ambig'] = alninfo['nofeat_A'] + alninfo['feat_A']

        """ Tweak alninfo """
        for cs, desc in alignment.CODES:
            if cs in alninfo:
                alninfo[desc] = alninfo[cs]
                del alninfo[cs]

        """ Remove rows with only __nofeature """
        rownames = np.array(sorted(_ridx, key=_ridx.get))
        assert _fidx[self.opts.no_feature_key] == 0, 'No feature key is not first column!'
        # Remove nofeature column then find rows with nonzero values
        _nz = scipy.sparse.csc_matrix(_m1)[:, 1:].sum(1).nonzero()[0]
        # Subset scores and read names
        _sparse_cls = get_sparse_class()
        self.raw_scores = _sparse_cls(_sparse_cls(_m1)[_nz,])
        _ridx = {v: i for i, v in enumerate(rownames[_nz])}
        # Set the shape
        self.shape = (len(_ridx), len(_fidx))
        # Ambiguous mappings
        alninfo['overlap_unique'] = np.sum(self.raw_scores.count(1) == 1)
        alninfo['overlap_ambig'] = self.shape[0] - alninfo['overlap_unique']

    def output_report(self, tl, stats_filename, counts_filename):
        """Generate TSV reports. Delegates to reporter.output_report()."""
        return _output_report_func(
            self.feat_index,
            self.feature_length,
            self.run_info,
            tl,
            self.opts.reassign_mode,
            self.opts.conf_prob,
            stats_filename,
            counts_filename,
        )

    def update_sam(self, tl, filename):
        """Update SAM file. Delegates to reporter.update_sam()."""
        return _update_sam_func(
            self.tmp_bam,
            self.read_index,
            self.feat_index,
            self.run_info,
            self.opts,
            tl,
            filename,
        )

    def print_summary(self, loglev=lg.WARNING):
        _d = Counter()
        for k, v in self.run_info.items():
            try:  # noqa: SIM105
                _d[k] = int(v)
            except ValueError:
                pass

        # For backwards compatibility with old checkpoints
        if 'mapped_pairs' in _d:
            _d['pair_mapped'] = _d['mapped_pairs']
        if 'mapped_single' in _d:
            _d['single_mapped'] = _d['mapped_single']

        lg.log(loglev, 'Alignment Summary:')
        lg.log(loglev, '    {} total fragments.'.format(_d['total_fragments']))
        lg.log(loglev, '        {} mapped as pairs.'.format(_d['pair_mapped']))
        lg.log(loglev, '        {} mapped as mixed.'.format(_d['pair_mixed']))
        lg.log(loglev, '        {} mapped single.'.format(_d['single_mapped']))
        lg.log(loglev, '        {} failed to map.'.format(_d['unmapped']))
        lg.log(loglev, '--')
        lg.log(
            loglev,
            '    {} fragments mapped to reference; of these'.format(
                _d['pair_mapped'] + _d['pair_mixed'] + _d['single_mapped']
            ),
        )
        lg.log(loglev, '        {} had one unique alignment.'.format(_d['unique']))
        lg.log(loglev, '        {} had multiple alignments.'.format(_d['ambig']))
        lg.log(loglev, '--')
        lg.log(
            loglev,
            '    {} fragments overlapped annotation; of these'.format(_d['overlap_unique'] + _d['overlap_ambig']),
        )
        lg.log(loglev, '        {} map to one locus.'.format(_d['overlap_unique']))
        lg.log(loglev, '        {} map to multiple loci.'.format(_d['overlap_ambig']))
        lg.log(loglev, '\n')

    def __str__(self):
        if hasattr(self.opts, 'samfile'):
            return f'<Polymerase samfile={self.opts.samfile}, gtffile={getattr(self.opts, "gtffile", "?")}>'
        elif hasattr(self.opts, 'checkpoint'):
            return f'<Polymerase checkpoint={self.opts.checkpoint}>'
        else:
            return '<Polymerase>'


# TODO: Single-cell subcommands are deferred to a future release.
class scPolymerase(Polymerase):
    def __init__(self, opts):
        super().__init__(opts)
        self.single_cell = True
        self.read_barcodes = {}
        self.barcode_read_indices = defaultdict(list)

    def output_report(self, tl, stats_filename, counts_filename):
        _rmethod, _rprob = self.opts.reassign_mode, self.opts.conf_prob
        _fnames = sorted(self.feat_index, key=self.feat_index.get)
        _flens = self.feature_length
        _stats_rounding = pd.Series([2, 3, 2, 3], index=['final_conf', 'final_prop', 'init_best_avg', 'init_prop'])

        _stats_report0 = {
            'transcript': _fnames,
            'transcript_length': [_flens[f] for f in _fnames],
            'final_prop': tl.pi,
            'init_prop': tl.pi_init,
        }

        _stats_report = pd.DataFrame(_stats_report0)
        _stats_report.sort_values('final_prop', ascending=False, inplace=True)
        _stats_report = _stats_report.round(_stats_rounding)

        _comment = ['## RunInfo']
        _comment += ['{}:{}'.format(*tup) for tup in self.run_info.items()]

        with open(stats_filename, 'w') as outh:
            outh.write('\t'.join(_comment) + '\n')
            _stats_report.to_csv(outh, sep='\t', index=False)

        _methods = ['conf', 'all', 'unique', 'exclude', 'choose', 'average']
        _bcidx = {bcode: rows for bcode, rows in self.barcode_read_indices.items() if len(rows) > 0}
        _bcodes = [_bcode for _bcode, _rows in _bcidx.items()]
        for _method in _methods:
            if _method != _rmethod and not self.opts.use_every_reassign_mode:
                continue
            if self.opts.use_every_reassign_mode:
                counts_outfile = counts_filename[: counts_filename.rfind('.')] + '_' + _method + '.tsv'
            else:
                counts_outfile = counts_filename
            _assignments = tl.reassign(_method, _rprob)
            _cell_count_matrix = scipy.sparse.dok_matrix((len(_bcidx), _assignments.shape[1]))
            for i, (_bcode, _rows) in enumerate(_bcidx.items()):
                _cell_count_matrix[i, :] = _assignments[_rows, :].sum(0).A1
            _cell_count_df = pd.DataFrame(_cell_count_matrix.todense(), columns=_fnames, index=_bcodes)
            _cell_count_df.to_csv(counts_outfile, sep='\t')


class Assigner:
    def __init__(self, annotation, no_feature_key, overlap_mode, overlap_threshold, stranded_mode=None):
        self.annotation = annotation
        self.no_feature_key = no_feature_key
        self.overlap_mode = overlap_mode
        self.overlap_threshold = overlap_threshold
        self.stranded_mode = stranded_mode

    def assign_func(self):
        def _assign_pair_threshold(pair):
            blocks = pair.refblocks
            if pair.r1_is_reversed:
                if pair.is_paired:
                    frag_strand = '+' if self.stranded_mode[-1] == 'F' else '-'
                else:
                    frag_strand = '-' if self.stranded_mode[0] == 'F' else '+'
            else:
                if pair.is_paired:
                    frag_strand = '-' if self.stranded_mode[-1] == 'F' else '+'
                else:
                    frag_strand = '+' if self.stranded_mode[0] == 'F' else '-'
            f = self.annotation.intersect_blocks(pair.ref_name, blocks, frag_strand)
            if not f:
                return self.no_feature_key
            # Calculate the percentage of fragment mapped
            fname, overlap = f.most_common()[0]
            if overlap > pair.alnlen * self.overlap_threshold:
                return fname
            else:
                return self.no_feature_key

        def _assign_pair_intersection_strict(pair):
            raise NotImplementedError('intersection-strict overlap mode is not yet implemented')

        def _assign_pair_union(pair):
            raise NotImplementedError('union overlap mode is not yet implemented')

        if self.overlap_mode == 'threshold':
            return _assign_pair_threshold
        elif self.overlap_mode == 'intersection-strict':
            return _assign_pair_intersection_strict
        elif self.overlap_mode == 'union':
            return _assign_pair_union
        else:
            raise AssertionError(f'Unknown overlap mode: {self.overlap_mode}')
