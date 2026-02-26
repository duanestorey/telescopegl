# -*- coding: utf-8 -*-

# This file is part of Polymerase.
# Original Telescope code by Matthew L. Bendall (https://github.com/mlbendall/telescope)
#
# New code and modifications by Duane Storey (https://github.com/duanestorey) and Claude (Anthropic).
# Licensed under MIT License.

"""Backend-agnostic compute operations for Polymerase primers.

Primer authors call ``get_ops()`` to get a :class:`ComputeOps` instance,
then use its methods instead of calling scipy/cupy directly.  The active
backend handles dispatch automatically:

- **CpuOps** — scipy.sparse + numpy (always available)
- **GpuOps** — CuPy + cuSPARSE (when ``cupy`` and CUDA are present)

Example::

    from polymerase.compute import get_ops

    ops = get_ops()
    normalised = ops.norm(matrix, axis=1)
    result = ops.dot(normalised, weights)
"""

from .ops import ComputeOps  # noqa: F401


def get_ops() -> ComputeOps:
    """Return a :class:`ComputeOps` for the active backend.

    GPU ops are returned when CuPy + CUDA are available; otherwise CPU ops.
    """
    from ..sparse.backend import get_backend

    backend = get_backend()
    if getattr(backend, 'has_cupy', False):
        from .gpu_ops import GpuOps
        return GpuOps()
    from .cpu_ops import CpuOps
    return CpuOps()
