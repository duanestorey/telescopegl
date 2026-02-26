# This file is part of Polymerase.
# Original Telescope code by Matthew L. Bendall (https://github.com/mlbendall/telescope)
#
# New code and modifications by Duane Storey (https://github.com/duanestorey) and Claude (Anthropic).
# Licensed under MIT License.


import logging as lg
from collections import Counter, OrderedDict

import HTSeq


class _AnnotationHTSeq:
    def __init__(self, gtf_file, attribute_name):
        lg.debug('Using HTSeq for annotation.')
        self.loci = OrderedDict()
        self.features = HTSeq.GenomicArrayOfSets('auto', stranded=False)
        for f in HTSeq.GFF_Reader(gtf_file, end_included=True):
            if f.type == 'exon':
                self.features[f.iv] += f.attr[attribute_name]
                if f.attr[attribute_name] not in self.loci:
                    self.loci[f.attr[attribute_name]] = list()
                self.loci[f.attr[attribute_name]].append(f)

    def feature_length(self):
        """Get feature lengths

        Returns:
            (dict of str: int): Feature names to feature lengths

        """
        ret = Counter()
        for ref_iv, step_set in self.features.steps():
            for feat_id in step_set:
                ret[feat_id] += ref_iv.length
        return ret

    def intersect_blocks(self, ref, blocks):
        _result = Counter()
        for b_start, b_end in blocks:
            iv = HTSeq.GenomicInterval(ref, b_start, b_end)
            for ref_iv, step_set in self.features[iv].steps():
                for feat_id in step_set:
                    _result[feat_id] += ref_iv.length
        return _result
