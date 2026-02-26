# -*- coding: utf-8 -*-

# This file is part of Polymerase.
# Original Telescope code by Matthew L. Bendall (https://github.com/mlbendall/telescope)
#
# New code and modifications by Duane Storey (https://github.com/duanestorey) and Claude (Anthropic).
# Licensed under MIT License.

"""Plugin discovery, loading, and lifecycle management."""

import logging as lg
from importlib.metadata import entry_points

from .abc import Primer, Cofactor


class PluginRegistry:
    """Discovers, configures, and manages primers and cofactors.

    Plugins are discovered via ``importlib.metadata.entry_points`` using the
    groups ``polymerase.primers`` and ``polymerase.cofactors``.
    """

    def __init__(self):
        self._primers: list[Primer] = []
        self._cofactors: list[Cofactor] = []
        self._all_primer_eps = {}   # name -> entry_point
        self._all_cofactor_eps = {}

    # -- Discovery -----------------------------------------------------------

    def discover(self, active_primers=None):
        """Load primer and cofactor entry-points.

        Args:
            active_primers: Iterable of primer names to activate, or *None*
                for all available primers.
        """
        # Discover all available primers
        primer_eps = entry_points(group='polymerase.primers')
        for ep in primer_eps:
            self._all_primer_eps[ep.name] = ep

        # Discover all available cofactors
        cofactor_eps = entry_points(group='polymerase.cofactors')
        for ep in cofactor_eps:
            self._all_cofactor_eps[ep.name] = ep

        # Determine which primers to load
        if active_primers is None:
            names_to_load = list(self._all_primer_eps.keys())
        else:
            names_to_load = list(active_primers)

        # Load primers
        for name in names_to_load:
            if name not in self._all_primer_eps:
                lg.warning("Primer '{}' not found, skipping".format(name))
                continue
            try:
                cls = self._all_primer_eps[name].load()
                instance = cls()
                if not isinstance(instance, Primer):
                    lg.warning("'{}' is not a Primer subclass, skipping".format(name))
                    continue
                self._primers.append(instance)
                lg.info("Loaded primer: {} v{}".format(instance.name, instance.version))
            except Exception as exc:
                lg.warning("Failed to load primer '{}': {}".format(name, exc))

        # Load cofactors whose primer is active
        active_primer_names = {p.name for p in self._primers}
        for name, ep in self._all_cofactor_eps.items():
            try:
                cls = ep.load()
                instance = cls()
                if not isinstance(instance, Cofactor):
                    lg.warning("'{}' is not a Cofactor subclass, skipping".format(name))
                    continue
                if instance.primer_name not in active_primer_names:
                    lg.debug("Cofactor '{}' skipped (primer '{}' not active)".format(
                        name, instance.primer_name))
                    continue
                self._cofactors.append(instance)
                lg.info("Loaded cofactor: {} (for primer '{}')".format(
                    instance.name, instance.primer_name))
            except Exception as exc:
                lg.warning("Failed to load cofactor '{}': {}".format(name, exc))

    # -- Configuration -------------------------------------------------------

    def configure_all(self, opts, compute_ops):
        """Pass CLI options and compute ops to all loaded plugins."""
        for p in self._primers:
            try:
                p.configure(opts, compute_ops)
            except Exception as exc:
                lg.warning("Primer '{}' configure failed: {}".format(p.name, exc))

        for c in self._cofactors:
            try:
                c.configure(opts, compute_ops)
            except Exception as exc:
                lg.warning("Cofactor '{}' configure failed: {}".format(c.name, exc))

    # -- Hook notification ---------------------------------------------------

    def notify(self, hook_name, snapshot):
        """Fire a hook on all active primers.

        Args:
            hook_name: Method name, e.g. ``'on_annotation_loaded'``.
            snapshot: Frozen dataclass to pass.
        """
        for p in self._primers:
            method = getattr(p, hook_name, None)
            if method is None:
                continue
            try:
                method(snapshot)
            except Exception as exc:
                lg.warning("Primer '{}' {} failed: {}".format(
                    p.name, hook_name, exc))

    # -- Commit & transform --------------------------------------------------

    def commit_all(self, output_dir, exp_tag):
        """Call commit() on every primer, then run each primer's cofactors."""
        import os

        for p in self._primers:
            primer_dir = output_dir  # default primer writes to top-level
            try:
                lg.info("Committing primer: {}".format(p.name))
                p.commit(primer_dir, exp_tag)
            except Exception as exc:
                lg.warning("Primer '{}' commit failed: {}".format(p.name, exc))
                continue

            # Run cofactors for this primer
            for c in self._cofactors:
                if c.primer_name != p.name:
                    continue
                cofactor_dir = os.path.join(output_dir, c.name)
                os.makedirs(cofactor_dir, exist_ok=True)
                try:
                    lg.info("Running cofactor: {} (for {})".format(c.name, p.name))
                    c.transform(primer_dir, cofactor_dir, exp_tag)
                except Exception as exc:
                    lg.warning("Cofactor '{}' transform failed: {}".format(
                        c.name, exc))

    # -- Introspection -------------------------------------------------------

    def list_available(self):
        """Return dicts of all discoverable primers and cofactors."""
        primers = {}
        for name, ep in self._all_primer_eps.items():
            try:
                cls = ep.load()
                inst = cls()
                primers[name] = {
                    'version': inst.version,
                    'description': inst.description,
                    'builtin': ep.value.startswith('polymerase.plugins.builtin'),
                }
            except Exception:
                primers[name] = {'version': '?', 'description': '(load failed)', 'builtin': False}

        cofactors = {}
        for name, ep in self._all_cofactor_eps.items():
            try:
                cls = ep.load()
                inst = cls()
                cofactors[name] = {
                    'primer_name': inst.primer_name,
                    'description': inst.description,
                    'builtin': ep.value.startswith('polymerase.plugins.builtin'),
                }
            except Exception:
                cofactors[name] = {'primer_name': '?', 'description': '(load failed)', 'builtin': False}

        return primers, cofactors

    @property
    def primers(self):
        return list(self._primers)

    @property
    def cofactors(self):
        return list(self._cofactors)
