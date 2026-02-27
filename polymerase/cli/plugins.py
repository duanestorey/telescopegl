# This file is part of Polymerase.
# Original Telescope code by Matthew L. Bendall (https://github.com/mlbendall/telescope)
#
# New code and modifications by Duane Storey (https://github.com/duanestorey) and Claude (Anthropic).
# Licensed under MIT License.

"""CLI subcommands for plugin management."""

import re
import subprocess
import sys

# PEP 508 package name: letters, digits, hyphens, underscores, dots
_VALID_PACKAGE_RE = re.compile(
    r'^[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?(\[.*\])?(==|>=|<=|!=|~=|>|<)?[A-Za-z0-9.*]*$'
)


def list_plugins(args):
    """List all installed primers and cofactors."""
    from ..plugins.registry import PluginRegistry

    registry = PluginRegistry()
    registry.discover()
    primers, cofactors = registry.list_available()

    print('Primers:')
    if not primers:
        print('  (none)')
    for name, info in sorted(primers.items()):
        tag = ' (built-in)' if info.get('builtin') else ''
        desc = info.get('description', '')
        ver = info.get('version', '?')
        print(f'  {name:20s} v{ver:<10s} {desc}{tag}')

    print()
    print('Cofactors:')
    if not cofactors:
        print('  (none)')
    for name, info in sorted(cofactors.items()):
        tag = ' (built-in)' if info.get('builtin') else ''
        desc = info.get('description', '')
        primer = info.get('primer_name', '?')
        print(f'  {name:20s} for {primer:10s} {desc}{tag}')


def install_plugin(args):
    """Install a primer/cofactor package via pip."""
    package = args.package
    if not _VALID_PACKAGE_RE.match(package):
        print(f'Error: invalid package name: {package!r}')
        sys.exit(1)
    print(f'Installing {package}...')
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', package],
        capture_output=False,
    )
    if result.returncode == 0:
        print("\nInstalled. Run 'polymerase list-plugins' to verify.")
    else:
        print(f'\nInstallation failed (exit code {result.returncode}).')
    sys.exit(result.returncode)
