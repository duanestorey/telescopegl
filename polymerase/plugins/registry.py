# This file is part of Polymerase.
# Original Telescope code by Matthew L. Bendall (https://github.com/mlbendall/telescope)
#
# New code and modifications by Duane Storey (https://github.com/duanestorey) and Claude (Anthropic).
# Licensed under MIT License.

"""Plugin discovery, loading, and lifecycle management."""

import logging as lg
from importlib.metadata import entry_points

from .abc import Cofactor, Primer


class PluginRegistry:
    """Discovers, configures, and manages primers and cofactors.

    Plugins are discovered via ``importlib.metadata.entry_points`` using the
    groups ``polymerase.primers`` and ``polymerase.cofactors``.
    """

    def __init__(self):
        self._primers: list[Primer] = []
        self._cofactors: list[Cofactor] = []
        self._all_primer_eps = {}  # name -> entry_point
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
                lg.warning(f"Primer '{name}' not found, skipping")
                continue
            try:
                cls = self._all_primer_eps[name].load()
                instance = cls()
                if not isinstance(instance, Primer):
                    lg.warning(f"'{name}' is not a Primer subclass, skipping")
                    continue
                self._primers.append(instance)
                lg.info(f'Loaded primer: {instance.name} v{instance.version}')
            except Exception as exc:
                lg.warning(f"Failed to load primer '{name}': {exc}", exc_info=True)

        # Load cofactors whose primer is active
        active_primer_names = {p.name for p in self._primers}
        for name, ep in self._all_cofactor_eps.items():
            try:
                cls = ep.load()
                instance = cls()
                if not isinstance(instance, Cofactor):
                    lg.warning(f"'{name}' is not a Cofactor subclass, skipping")
                    continue
                if instance.primer_name not in active_primer_names:
                    lg.debug(f"Cofactor '{name}' skipped (primer '{instance.primer_name}' not active)")
                    continue
                self._cofactors.append(instance)
                lg.info(f"Loaded cofactor: {instance.name} (for primer '{instance.primer_name}')")
            except Exception as exc:
                lg.warning(f"Failed to load cofactor '{name}': {exc}", exc_info=True)

    # -- Configuration -------------------------------------------------------

    def configure_all(self, opts, compute_ops):
        """Pass CLI options and compute ops to all loaded plugins."""
        for p in self._primers:
            try:
                p.configure(opts, compute_ops)
            except Exception as exc:
                lg.warning(f"Primer '{p.name}' configure failed: {exc}", exc_info=True)

        for c in self._cofactors:
            try:
                c.configure(opts, compute_ops)
            except Exception as exc:
                lg.warning(f"Cofactor '{c.name}' configure failed: {exc}", exc_info=True)

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
                lg.warning(f"Primer '{p.name}' {hook_name} failed: {exc}", exc_info=True)

    # -- Commit & transform --------------------------------------------------

    def commit_all(self, output_dir, exp_tag, console=None, stopwatch=None):
        """Call commit() on every primer, then run each primer's cofactors."""
        import os

        for p in self._primers:
            primer_dir = output_dir  # default primer writes to top-level
            if console:
                console.section(f'Running {p.name}...')
            try:
                lg.info(f'Committing primer: {p.name}')
                p.commit(primer_dir, exp_tag, console=console, stopwatch=stopwatch)
            except Exception as exc:
                lg.warning(f"Primer '{p.name}' commit failed: {exc}", exc_info=True)
                continue

            # Run cofactors for this primer
            _primer_cofactors = [c for c in self._cofactors if c.primer_name == p.name]
            if _primer_cofactors:
                if stopwatch:
                    stopwatch.start('Cofactors')
                if console:
                    console.blank()
                    console.section('Running cofactors...')
            for c in _primer_cofactors:
                cofactor_dir = os.path.join(output_dir, c.name)
                os.makedirs(cofactor_dir, exist_ok=True)
                try:
                    lg.info(f'Running cofactor: {c.name} (for {p.name})')
                    if stopwatch:
                        stopwatch.start(c.name, level=1)
                    c.transform(primer_dir, cofactor_dir, exp_tag, console=console, stopwatch=stopwatch)
                    if stopwatch:
                        stopwatch.stop()
                except Exception as exc:
                    if stopwatch:
                        stopwatch.stop()
                    lg.warning(f"Cofactor '{c.name}' transform failed: {exc}", exc_info=True)
            if _primer_cofactors and stopwatch:
                # Stop the top-level Cofactors timer if sub-stages didn't
                # already consume it (stop is safe to call when inactive)
                stopwatch.stop()

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
