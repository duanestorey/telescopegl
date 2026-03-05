# This file is part of Polymerase.
# Original Telescope code by Matthew L. Bendall (https://github.com/mlbendall/telescope)
#
# New code and modifications by Duane Storey (https://github.com/duanestorey) and Claude (Anthropic).
# Licensed under MIT License.

import os

from polymerase.annotation import get_annotation_class
from polymerase.annotation.gtf_utils import detect_family_class_attrs
from polymerase.tests import TEST_DATA_DIR

# Path to bundled annotation with gene + exon lines
BUNDLED_GTF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'annotation.gtf')


class TestAnnotationIntervalTree:
    @classmethod
    def setup_class(cls):
        cls.AnnotationClass = get_annotation_class('intervaltree')

    @classmethod
    def teardown_class(cls):
        del cls.AnnotationClass

    def setup_method(self):
        self.gtffile = os.path.join(TEST_DATA_DIR, 'annotation_test.2.gtf')
        self.A = self.AnnotationClass(self.gtffile, 'locus', None)

    def teardown_method(self):
        del self.A

    def test_correct_type(self):
        assert type(self.A) is get_annotation_class('intervaltree')

    def test_annot_created(self):
        assert self.A.key == 'locus'

    def test_annot_treesize(self):
        assert len(self.A.itree['chr1']) == 3
        assert len(self.A.itree['chr2']) == 4
        assert len(self.A.itree['chr3']) == 2

    def test_empty_lookups(self):
        assert not self.A.intersect_blocks('chr1', [(1, 9999)])
        assert not self.A.intersect_blocks('chr1', [(20001, 39999)])
        assert not self.A.intersect_blocks('chr1', [(50001, 79999)])
        assert not self.A.intersect_blocks('chr1', [(90001, 90001)])
        assert not self.A.intersect_blocks('chr1', [(190000, 590000)])
        assert not self.A.intersect_blocks('chr2', [(1, 9999)])
        assert not self.A.intersect_blocks('chr3', [(1, 9999)])
        assert not self.A.intersect_blocks('chr4', [(1, 1000000000)])
        assert not self.A.intersect_blocks('chrX', [(1, 1000000000)])

    def test_simple_lookups(self):
        lines = (l.strip('\n').split('\t') for l in open(self.gtffile))
        for l in lines:
            iv = (int(l[3]), int(l[4]))
            loc = l[8].split('"')[1]
            r = self.A.intersect_blocks(l[0], [iv])
            assert loc in r
            assert (r[loc] - 1) == (iv[1] - iv[0]), f'{r[loc]} not equal to {iv[1] - iv[0]}'

    def test_overlap_lookups(self):
        assert self.A.intersect_blocks('chr1', [(1, 10000)])['locus1'] == 1
        assert self.A.intersect_blocks('chr2', [(1, 10000)])['locus4'] == 1
        assert self.A.intersect_blocks('chr3', [(1, 10000)])['locus7'] == 1
        r = self.A.intersect_blocks('chr1', [(19990, 40000)])
        assert r['locus1'] == 11 and r['locus2'] == 1
        r = self.A.intersect_blocks('chr2', [(44990, 46010)])
        assert r['locus5'] == 22
        r = self.A.intersect_blocks('chr3', [(44990, 46010)])
        assert r['locus8'] == 1021

    def test_subregion_chrom(self):
        sA = self.A.subregion('chr3')
        assert not sA.intersect_blocks('chr1', [(1, 10000)])
        assert not sA.intersect_blocks('chr2', [(1, 10000)])
        assert sA.intersect_blocks('chr3', [(1, 10000)])['locus7'] == 1
        r = sA.intersect_blocks('chr3', [(44990, 46010)])
        assert r['locus8'] == 1021

    def test_subregion_reg(self):
        sA = self.A.subregion('chr3', 30000, 50000)
        assert not sA.intersect_blocks('chr1', [(1, 10000)])
        assert not sA.intersect_blocks('chr2', [(1, 10000)])
        assert not sA.intersect_blocks('chr3', [(1, 10000)])
        assert sA.intersect_blocks('chr3', [(40000, 45000)])['locus8'] == 5001
        r = sA.intersect_blocks('chr3', [(44990, 46010)])
        assert r['locus8'] == 1021


class TestMultiFeatureType:
    """Tests for --feature_type support with multiple GTF feature types."""

    @classmethod
    def setup_class(cls):
        cls.AnnotationClass = get_annotation_class('intervaltree')
        cls.mixed_gtf = os.path.join(TEST_DATA_DIR, 'annotation_test.3.gtf')

    def test_default_exon_only(self):
        """Default feature_type='exon' loads only exon lines."""
        annot = self.AnnotationClass(self.mixed_gtf, 'locus', None)
        assert len(annot.loci) == 4  # locus1-4 have exon lines

    def test_gene_only(self):
        """feature_type='gene' loads only gene lines."""
        annot = self.AnnotationClass(self.mixed_gtf, 'locus', None, feature_type='gene')
        assert len(annot.loci) == 4  # locus1-4 have gene lines

    def test_exon_and_gene(self):
        """feature_type={'exon', 'gene'} loads both types."""
        annot = self.AnnotationClass(self.mixed_gtf, 'locus', None, feature_type={'exon', 'gene'})
        assert len(annot.loci) == 4  # same loci, but covered by both types

    def test_exon_gene_and_transcript(self):
        """feature_type with transcript includes locus5."""
        annot = self.AnnotationClass(self.mixed_gtf, 'locus', None, feature_type={'exon', 'gene', 'transcript'})
        assert len(annot.loci) == 5  # locus5 only has transcript line
        assert 'locus5' in annot.loci

    def test_transcript_only(self):
        """feature_type='transcript' loads only the transcript line."""
        annot = self.AnnotationClass(self.mixed_gtf, 'locus', None, feature_type='transcript')
        assert len(annot.loci) == 1
        assert 'locus5' in annot.loci

    def test_comma_separated_string(self):
        """Comma-separated string is parsed correctly."""
        annot = self.AnnotationClass(self.mixed_gtf, 'locus', None, feature_type='exon,gene')
        assert len(annot.loci) == 4

    def test_frozenset_accepted(self):
        """frozenset is accepted as feature_type."""
        annot = self.AnnotationClass(self.mixed_gtf, 'locus', None, feature_type=frozenset(['exon', 'transcript']))
        assert len(annot.loci) == 5

    def test_list_accepted(self):
        """list is accepted as feature_type."""
        annot = self.AnnotationClass(self.mixed_gtf, 'locus', None, feature_type=['exon'])
        assert len(annot.loci) == 4

    def test_bundled_gtf_default_exon(self):
        """Bundled annotation.gtf: default loads 99 loci (exon lines)."""
        if not os.path.exists(BUNDLED_GTF):
            import pytest

            pytest.skip('Bundled GTF not available')
        annot = self.AnnotationClass(BUNDLED_GTF, 'locus', None)
        assert len(annot.loci) == 99

    def test_bundled_gtf_exon_and_gene(self):
        """Bundled annotation.gtf: exon+gene loads more features (gene lines add loci)."""
        if not os.path.exists(BUNDLED_GTF):
            import pytest

            pytest.skip('Bundled GTF not available')
        annot_exon = self.AnnotationClass(BUNDLED_GTF, 'locus', None, feature_type='exon')
        annot_both = self.AnnotationClass(BUNDLED_GTF, 'locus', None, feature_type={'exon', 'gene'})
        # gene lines should not add new loci (same locus attribute) but they are loaded
        assert len(annot_both.loci) >= len(annot_exon.loci)


class TestDetectFamilyClassAttrsFeatureTypes:
    """Test that detect_family_class_attrs respects feature_types."""

    def test_default_uses_exon(self):
        if not os.path.exists(BUNDLED_GTF):
            import pytest

            pytest.skip('Bundled GTF not available')
        fam, cls = detect_family_class_attrs(BUNDLED_GTF)
        assert fam == 'repFamily'
        assert cls == 'repClass'

    def test_explicit_exon(self):
        if not os.path.exists(BUNDLED_GTF):
            import pytest

            pytest.skip('Bundled GTF not available')
        fam, cls = detect_family_class_attrs(BUNDLED_GTF, feature_types={'exon'})
        assert fam == 'repFamily'
        assert cls == 'repClass'

    def test_gene_feature_type(self):
        if not os.path.exists(BUNDLED_GTF):
            import pytest

            pytest.skip('Bundled GTF not available')
        fam, cls = detect_family_class_attrs(BUNDLED_GTF, feature_types={'gene'})
        assert fam == 'repFamily' or fam == 'family_id'
