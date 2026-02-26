# -*- coding: utf-8 -*-

# This file is part of Polymerase.
# Original Telescope code by Matthew L. Bendall (https://github.com/mlbendall/telescope)
#
# New code and modifications by Duane Storey (https://github.com/duanestorey) and Claude (Anthropic).
# Licensed under MIT License.

"""Backend abstraction for Polymerase compute acceleration.

Provides auto-detection of available acceleration libraries (Numba, MKL)
and returns the appropriate sparse matrix class for the active backend.
"""
import logging as lg
from dataclasses import dataclass, field


@dataclass
class BackendInfo:
    """Information about the active compute backend."""
    name: str = 'cpu_stock'
    has_numba: bool = False
    has_mkl: bool = False
    threadsafe_parallelism: bool = False


# Module-level singleton
_active_backend: BackendInfo = None


def _detect_numba():
    """Check if numba is available and functional."""
    try:
        import numba  # noqa: F401
        return True
    except ImportError:
        return False


def _detect_threadsafe_threading():
    """Check if Numba's threading layer supports concurrent access.

    The workqueue layer (default) is NOT threadsafe and will crash if
    Numba parallel functions are called from multiple Python threads.
    TBB and OpenMP layers are threadsafe.
    """
    try:
        import numba
        layer = getattr(numba.config, 'THREADING_LAYER', 'default')
        # After JIT compilation, we can check the actual active layer
        # But before that, we check the configured preference
        if layer in ('tbb', 'omp'):
            return True
        # 'default' or 'workqueue' — not safe for concurrent access
        return False
    except (ImportError, AttributeError):
        return False


def _detect_mkl():
    """Check if sparse_dot_mkl is available."""
    try:
        import sparse_dot_mkl  # noqa: F401
        return True
    except ImportError:
        return False


def configure():
    """Configure the compute backend based on available libraries.

    Tries Numba+MKL first, falls back to stock scipy.
    """
    global _active_backend

    has_numba = _detect_numba()
    has_mkl = _detect_mkl()

    threadsafe = _detect_threadsafe_threading() if has_numba else False

    if has_numba:
        name = 'cpu_optimized'
        lg.info('Backend: cpu_optimized (numba={}, mkl={}, threadsafe={})'.format(
            has_numba, has_mkl, threadsafe))
    else:
        name = 'cpu_stock'
        lg.info('Backend: cpu_stock (scipy only)')

    _active_backend = BackendInfo(
        name=name,
        has_numba=has_numba,
        has_mkl=has_mkl,
        threadsafe_parallelism=threadsafe,
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
            from .numba_matrix import CpuOptimizedCsrMatrix
            return CpuOptimizedCsrMatrix
        except ImportError:
            pass
    from .matrix import csr_matrix_plus
    return csr_matrix_plus
