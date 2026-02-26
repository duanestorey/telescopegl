# -*- coding: utf-8 -*-

# This file is part of Polymerase.
# Original Telescope code by Matthew L. Bendall (https://github.com/mlbendall/telescope)
#
# New code and modifications by Duane Storey (https://github.com/duanestorey) and Claude (Anthropic).
# Licensed under MIT License.

"""Pretty stdout output for the Polymerase CLI.

Separate from Python logging (which goes to stderr). The Console writes
structured, human-readable progress to stdout while logging continues to
handle debug/diagnostic output on stderr.
"""

import sys


class Console:
    """Pretty stdout output for the Polymerase CLI."""

    QUIET = 0
    NORMAL = 1
    VERBOSE = 2
    DEBUG = 3

    def __init__(self, level=NORMAL, stream=None):
        self.level = level
        self.stream = stream or sys.stdout
        self._use_color = hasattr(self.stream, 'isatty') and self.stream.isatty()

    def banner(self, version):
        """Print product banner."""
        if self.level < self.NORMAL:
            return
        self._write('')
        if self._use_color:
            self._write(
                '\033[1mPolymerase v{}\033[0m'
                ' \u2014 Transposable Element Expression'.format(version))
        else:
            self._write(
                'Polymerase v{}'
                ' -- Transposable Element Expression'.format(version))
        self._write('')

    def section(self, title):
        """Print indented section header."""
        if self.level < self.NORMAL:
            return
        self._write('  {}'.format(title))

    def item(self, label, value, indent=4):
        """Print key: value pair."""
        if self.level < self.NORMAL:
            return
        padding = ' ' * indent
        self._write('{}{:<14}{}'.format(padding, label + ':', value))

    def status(self, message):
        """Print a status message at normal level."""
        if self.level < self.NORMAL:
            return
        self._write('  {}'.format(message))

    def detail(self, message):
        """Print indented detail at normal level."""
        if self.level < self.NORMAL:
            return
        self._write('    {}'.format(message))

    def verbose(self, message):
        """Print only in verbose/debug mode."""
        if self.level < self.VERBOSE:
            return
        self._write('    {}'.format(message))

    def plugin(self, name, description, version=None):
        """Print loaded plugin line."""
        if self.level < self.NORMAL:
            return
        check = '\u2713' if self._use_color else '*'
        ver = ' v{}'.format(version) if version else ''
        self._write('    {} {}{} \u2014 {}'.format(check, name, ver, description))

    def output_file(self, path):
        """Print an output file path."""
        if self.level < self.NORMAL:
            return
        self._write('    {}'.format(path))

    def blank(self):
        """Print a blank line."""
        if self.level < self.NORMAL:
            return
        self._write('')

    def _write(self, text):
        print(text, file=self.stream)
