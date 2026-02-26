# -*- coding: utf-8 -*-
"""Backend abstraction for Telescope compute acceleration.

Provides auto-detection of available acceleration libraries (Numba, MKL)
and returns the appropriate sparse matrix class for the active backend.
"""
import logging as lg
from dataclasses import dataclass, field

__author__ = 'Duane Storey'


@dataclass
class BackendInfo:
    """Information about the active compute backend."""
    name: str = 'cpu_stock'
    has_numba: bool = False
    has_mkl: bool = False
    has_gpu: bool = False


# Module-level singleton
_active_backend: BackendInfo = None


def _detect_numba():
    """Check if numba is available and functional."""
    try:
        import numba  # noqa: F401
        return True
    except ImportError:
        return False


def _detect_mkl():
    """Check if sparse_dot_mkl is available."""
    try:
        import sparse_dot_mkl  # noqa: F401
        return True
    except ImportError:
        return False


def configure(use_gpu=False):
    """Configure the compute backend based on available libraries.

    Tries Numba+MKL first, falls back to stock scipy.

    Args:
        use_gpu: Reserved for future GPU support.
    """
    global _active_backend

    has_numba = _detect_numba()
    has_mkl = _detect_mkl()

    if has_numba:
        name = 'cpu_optimized'
        lg.info('Backend: cpu_optimized (numba={}, mkl={})'.format(
            has_numba, has_mkl))
    else:
        name = 'cpu_stock'
        lg.info('Backend: cpu_stock (scipy only)')

    _active_backend = BackendInfo(
        name=name,
        has_numba=has_numba,
        has_mkl=has_mkl,
        has_gpu=False,
    )
    return _active_backend


def get_backend():
    """Return the active backend info. Defaults to cpu_stock if unconfigured."""
    global _active_backend
    if _active_backend is None:
        _active_backend = BackendInfo()
    return _active_backend


def get_sparse_class():
    """Return the appropriate sparse matrix class for the active backend.

    Returns:
        CpuOptimizedCsrMatrix if backend is cpu_optimized, else csr_matrix_plus.
    """
    backend = get_backend()
    if backend.name == 'cpu_optimized' and backend.has_numba:
        try:
            from .cpu_optimized_sparse import CpuOptimizedCsrMatrix
            return CpuOptimizedCsrMatrix
        except ImportError:
            pass
    from .sparse_plus import csr_matrix_plus
    return csr_matrix_plus
