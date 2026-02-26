# -*- coding: utf-8 -*-

# This file is part of Polymerase.
# Original Telescope code by Matthew L. Bendall (https://github.com/mlbendall/telescope)
#
# New code and modifications by Duane Storey (https://github.com/duanestorey) and Claude (Anthropic).
# Licensed under MIT License.

"""Abstract base classes for Polymerase plugins."""

from abc import ABC, abstractmethod


class Primer(ABC):
    """A primer processes platform data and produces output.

    The platform calls :meth:`on_annotation_loaded` and
    :meth:`on_matrix_built` with read-only snapshots.  The primer
    accumulates state internally, then :meth:`commit` writes results
    to disk.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in CLI and entry-point registration."""

    @property
    def description(self) -> str:
        """One-line human-readable description."""
        return ""

    @property
    def version(self) -> str:
        return "0.0.0"

    # -- Configuration -------------------------------------------------------

    def cli_options(self) -> str | None:
        """YAML string of extra CLI arguments, or *None*."""
        return None

    def configure(self, opts, compute) -> None:
        """Receive parsed CLI opts and a :class:`ComputeOps` instance."""

    # -- Platform hooks (receive frozen snapshots) ---------------------------

    def on_annotation_loaded(self, snapshot) -> None:
        """Called after GTF is parsed."""

    def on_matrix_built(self, snapshot) -> None:
        """Called after BAM → sparse matrix construction."""

    # -- Output --------------------------------------------------------------

    @abstractmethod
    def commit(self, output_dir: str, exp_tag: str) -> None:
        """Write results to *output_dir*."""


class Cofactor(ABC):
    """Transforms a primer's committed output."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier."""

    @property
    @abstractmethod
    def primer_name(self) -> str:
        """Name of the primer this cofactor acts on."""

    @property
    def description(self) -> str:
        return ""

    def cli_options(self) -> str | None:
        return None

    def configure(self, opts, compute) -> None:
        """Receive parsed CLI opts and a :class:`ComputeOps` instance."""

    @abstractmethod
    def transform(self, primer_output_dir: str, output_dir: str,
                  exp_tag: str) -> None:
        """Read the primer's output, write transformed results."""
