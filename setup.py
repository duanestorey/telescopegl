"""Setup polymerase package

Retained for Cython ext_modules support. Metadata is in pyproject.toml.
"""

from os import environ, path

from setuptools import Extension, setup

USE_CYTHON = True

CONDA_PREFIX = environ.get('CONDA_PREFIX', '.')
HTSLIB_INCLUDE_DIR = environ.get('HTSLIB_INCLUDE_DIR', None)

htslib_include_dirs = [
    HTSLIB_INCLUDE_DIR,
    path.join(CONDA_PREFIX, 'include'),
    path.join(CONDA_PREFIX, 'include', 'htslib'),
]
htslib_include_dirs = [d for d in htslib_include_dirs if path.exists(str(d))]

# Add pysam include dirs for cimport resolution and C headers
try:
    import pysam

    htslib_include_dirs.extend(pysam.get_include())
except (ImportError, IndexError):
    pass

ext = '.pyx' if USE_CYTHON else '.c'
extensions = [
    Extension(
        'polymerase.alignment.calignment',
        ['polymerase/alignment/calignment' + ext],
        include_dirs=htslib_include_dirs,
    ),
]

if USE_CYTHON:
    from Cython.Build import cythonize

    extensions = cythonize(extensions, include_path=['polymerase/alignment'] + htslib_include_dirs)

setup(
    ext_modules=extensions,
)
