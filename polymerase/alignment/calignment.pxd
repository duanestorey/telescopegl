from pysam.libcalignedsegment cimport AlignedSegment

cpdef list _merge_blocks_py(list ivs, int dist=*)

cdef class AlignedPair:
    cdef public AlignedSegment r1
    cdef public AlignedSegment r2
