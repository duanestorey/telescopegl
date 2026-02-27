# This file is part of Polymerase.
# Original Telescope code by Matthew L. Bendall (https://github.com/mlbendall/telescope)
#
# New code and modifications by Duane Storey (https://github.com/duanestorey) and Claude (Anthropic).
# Licensed under MIT License.


def get_annotation_class(annotation_class_name):
    """Get Annotation class matching provided name

    Args:
        annotation_class_name (str): Name of annotation class.

    Returns:
        Annotation class with data structure and function(s) for finding
        overlaps
    """
    if annotation_class_name == 'htseq':
        raise NotImplementedError('"htseq" is not compatible.')
        # from .htseq import _AnnotationHTSeq
        # return _AnnotationHTSeq
    elif annotation_class_name == 'intervaltree':
        from .intervaltree import _AnnotationIntervalTree

        return _AnnotationIntervalTree
    else:
        raise NotImplementedError(
            f'Unknown annotation class "{annotation_class_name}". Only "intervaltree" is supported.'
        )
