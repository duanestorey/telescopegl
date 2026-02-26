# -*- coding: utf-8 -*-

# This file is part of Polymerase.
# Original Telescope code by Matthew L. Bendall (https://github.com/mlbendall/telescope)
#
# New code and modifications by Duane Storey (https://github.com/duanestorey) and Claude (Anthropic).
# Licensed under MIT License.

"""CLI subcommands for plugin management."""

import subprocess
import sys


def list_plugins(args):
    """List all installed primers and cofactors."""
    from ..plugins.registry import PluginRegistry

    registry = PluginRegistry()
    registry.discover()
    primers, cofactors = registry.list_available()

    print("Primers:")
    if not primers:
        print("  (none)")
    for name, info in sorted(primers.items()):
        tag = " (built-in)" if info.get('builtin') else ""
        desc = info.get('description', '')
        ver = info.get('version', '?')
        print("  {:20s} v{:<10s} {}{}".format(name, ver, desc, tag))

    print()
    print("Cofactors:")
    if not cofactors:
        print("  (none)")
    for name, info in sorted(cofactors.items()):
        tag = " (built-in)" if info.get('builtin') else ""
        desc = info.get('description', '')
        primer = info.get('primer_name', '?')
        print("  {:20s} for {:10s} {}{}".format(name, primer, desc, tag))


def install_plugin(args):
    """Install a primer/cofactor package via pip."""
    package = args.package
    print("Installing {}...".format(package))
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', package],
        capture_output=False,
    )
    if result.returncode == 0:
        print("\nInstalled. Run 'polymerase list-plugins' to verify.")
    else:
        print("\nInstallation failed (exit code {}).".format(result.returncode))
    sys.exit(result.returncode)
