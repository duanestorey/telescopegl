# This file is part of Polymerase.
# Original Telescope code by Matthew L. Bendall (https://github.com/mlbendall/telescope)
#
# New code and modifications by Duane Storey (https://github.com/duanestorey) and Claude (Anthropic).
# Licensed under MIT License.

import logging as lg
import pickle
import re
from collections import Counter, OrderedDict, defaultdict, namedtuple

from intervaltree import Interval, IntervalTree

from .gtf_utils import detect_family_class_attrs

GTFRow = namedtuple('GTFRow', ['chrom', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute'])


def overlap_length(a, b):
    return max(0, min(a.end, b.end) - max(a.begin, b.begin))


def merge_intervals(a, b, d=None):
    return Interval(min(a.begin, b.begin), max(a.end, b.end), d)


class _AnnotationIntervalTree:
    def __init__(self, gtf_file, attribute_name, stranded_mode, feature_type='exon', classes=None):
        lg.debug('Using intervaltree for annotation.')
        self.loci = OrderedDict()
        self.key = attribute_name
        self.itree = defaultdict(IntervalTree)
        self.run_stranded = stranded_mode is not None and stranded_mode != 'None'

        # Auto-detect class attribute name for --classes filtering
        if classes and isinstance(gtf_file, str):
            _family_key, _class_key = detect_family_class_attrs(gtf_file)
            lg.info(f'Filtering GTF by classes: {classes} (using attribute: {_class_key})')
        else:
            _class_key = None

        # GTF filehandle
        _opened = isinstance(gtf_file, str)
        fh = open(gtf_file) if _opened else gtf_file  # noqa: SIM115
        try:
            for rownum, line in enumerate(fh):
                if line.startswith('#'):
                    continue
                f = GTFRow(*line.strip('\n').split('\t'))
                if f.feature != feature_type:
                    continue
                attr = dict(re.findall(r'(\w+)\s+"(.+?)";', f.attribute))
                attr['strand'] = f.strand

                # Filter by --classes if specified
                if classes and _class_key and attr.get(_class_key, '') not in classes:
                    continue

                if self.key not in attr:
                    lg.warning(f'Skipping row {rownum}: missing attribute "{self.key}"')
                    continue

                """ Add to locus list """
                if attr[self.key] not in self.loci:
                    self.loci[attr[self.key]] = list()
                self.loci[attr[self.key]].append(f)
                """ Add to interval tree """
                new_iv = Interval(int(f.start), int(f.end) + 1, attr)
                # Merge overlapping intervals from same locus
                overlap = self.itree[f.chrom].overlap(new_iv)
                if len(overlap) > 0:
                    mergeable = [iv for iv in overlap if iv.data[self.key] == attr[self.key]]
                    if mergeable:
                        for mergeable_iv in mergeable:
                            new_iv = merge_intervals(
                                mergeable_iv, new_iv, {self.key: attr[self.key], 'strand': attr['strand']}
                            )
                            self.itree[f.chrom].remove(mergeable_iv)
                self.itree[f.chrom].add(new_iv)
        finally:
            if _opened:
                fh.close()

    def feature_length(self):
        """Get feature lengths

        Returns:
            (dict of str: int): Feature names to feature lengths

        """
        ret = Counter()
        for chrom in list(self.itree.keys()):
            for iv in list(self.itree[chrom].items()):
                ret[iv.data[self.key]] += iv.length()
        return ret

    def subregion(self, ref, start_pos=None, end_pos=None):
        _subannot = type(self).__new__(type(self))
        _subannot.key = self.key
        _subannot.run_stranded = getattr(self, 'run_stranded', False)
        _subannot.itree = defaultdict(IntervalTree)

        if ref in self.itree:
            _subtree = self.itree[ref].copy()
            if start_pos is not None:
                _subtree.chop(_subtree.begin(), start_pos)
            if end_pos is not None:
                _subtree.chop(end_pos, _subtree.end() + 1)
            _subannot.itree[ref] = _subtree
        return _subannot

    def intersect_blocks(self, ref, blocks, frag_strand=None):
        _result = Counter()
        for b_start, b_end in blocks:
            query = Interval(b_start, (b_end + 1))
            for iv in self.itree[ref].overlap(query):
                if self.run_stranded:
                    if iv.data['strand'] == frag_strand:
                        _result[iv.data[self.key]] += overlap_length(iv, query)
                else:
                    _result[iv.data[self.key]] += overlap_length(iv, query)
        return _result

    def save(self, filename):
        with open(filename, 'wb') as outh:
            pickle.dump(
                {
                    'key': self.key,
                    'loci': self.loci,
                    'itree': self.itree,
                    'run_stranded': self.run_stranded,
                },
                outh,
            )

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as fh:
            loader = pickle.load(fh)
        obj = cls.__new__(cls)
        obj.key = loader['key']
        obj.loci = loader['loci']
        obj.itree = loader['itree']
        obj.run_stranded = loader.get('run_stranded', False)
        return obj
