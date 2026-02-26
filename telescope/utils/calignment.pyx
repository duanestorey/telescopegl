# -*- coding: utf-8 -*-
from telescope.utils.calignment cimport AlignedPair
from pysam.libcalignedsegment cimport AlignedSegment
from pysam.libcalignmentfile cimport AlignmentFile


__author__ = 'Matthew L. Bendall'
__copyright__ = "Copyright (C) 2019 Matthew L. Bendall"


from operator import itemgetter as _itemgetter
_key0 = _itemgetter(0)

cpdef list _merge_blocks_py(list ivs, int dist=0):
    """Merge overlapping or adjacent intervals."""
    if len(ivs) == 0:
        return []
    cdef list sivs = sorted(ivs, key=_key0)
    cdef list ret = [sivs[0]]
    cdef int s, e
    for iv in sivs[1:]:
        s = iv[0]
        e = iv[1]
        if s <= ret[-1][1] + dist:
            ret[-1] = (ret[-1][0], max(ret[-1][1], e))
        else:
            ret.append((s, e))
    return ret


cdef class AlignedPair:

    def __cinit__(self, AlignedSegment r1, AlignedSegment r2 = None):
        self.r1 = r1
        self.r2 = r2

    def __dealloc__(self):
        del self.r1
        del self.r2

    def write(self, outfile):
        """Write AlignedPair to file."""
        ret = outfile.write(self.r1)
        if self.r2:
            ret += outfile.write(self.r2)
        return ret

    def set_tag(self, tag, value, value_type=None, replace=True):
        self.r1.set_tag(tag, value, value_type, replace)
        if self.r2:
            self.r2.set_tag(tag, value, value_type, replace)

    def set_mapq(self, value):
        self.r1.mapping_quality = value
        if self.r2:
            self.r2.mapping_quality = value

    def set_flag(self, b):
        self.r1.flag = (self.r1.flag | b)
        if self.r2:
            self.r2.flag = (self.r2.flag | b)
        assert (self.r1.flag & b) == b

    def unset_flag(self, b):
        self.r1.flag = (self.r1.flag ^ (self.r1.flag & b))
        if self.r2:
            self.r2.flag = (self.r2.flag ^ (self.r2.flag & b))
        assert (self.r1.flag & b) == 0

    @property
    def numreads(self):
        return 1 if self.r2 is None else 2

    @property
    def is_paired(self):
        return self.r2 is not None

    @property
    def is_unmapped(self):
       return self.r1.is_unmapped

    @property
    def r1_is_reversed(self):
        return self.r1.is_reverse

    @property
    def ref_name(self):
        return self.r1.reference_name

    @property
    def query_id(self):
        return self.r1.query_name

    @property
    def refblocks(self):
        if self.r2 is None:
            return _merge_blocks_py(self.r1.get_blocks(), 1)
        else:
            blocks = self.r1.get_blocks() + self.r2.get_blocks()
            return _merge_blocks_py(blocks, 1)

    @property
    def alnlen(self):
        cdef int total = 0
        for b in self.refblocks:
            total += b[1] - b[0]
        return total

    @property
    def alnscore(self):
        if self.r2 is None:
            return self.r1.get_tag('AS')
        else:
            return self.r1.get_tag('AS') + self.r2.get_tag('AS')
