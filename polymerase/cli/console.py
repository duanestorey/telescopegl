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
from time import perf_counter


class Stopwatch:
    """Collects named timing segments for a benchmark summary."""

    def __init__(self):
        self._timings = []        # [(name, elapsed, level)]
        self._start = None
        self._active = None

    def start(self, name, level=0):
        """Begin timing a named stage. level=0 top-level, 1 sub-stage."""
        now = perf_counter()
        if self._active:
            self._timings.append(
                (self._active[0], now - self._active[1], self._active[2]))
        self._active = (name, now, level)
        if self._start is None:
            self._start = now

    def stop(self):
        """Stop the current segment."""
        if self._active:
            now = perf_counter()
            self._timings.append(
                (self._active[0], now - self._active[1], self._active[2]))
            self._active = None

    @property
    def total(self):
        return perf_counter() - self._start if self._start else 0.0

    @property
    def timings(self):
        return list(self._timings)


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

    def timing_table(self, stopwatch):
        """Print timing summary table."""
        if self.level < self.NORMAL:
            return
        timings = stopwatch.timings
        total = stopwatch.total
        if not timings:
            return

        self.section('Timing')
        for name, elapsed, level in timings:
            if level > 0 and self.level < self.VERBOSE:
                continue
            indent = '      ' if level > 0 else '    '
            if level == 0 and total > 0:
                pct = '{:>4.0f}%'.format(elapsed / total * 100)
            else:
                pct = ''
            self._write('{}{:<18}{:>5.1f}s{:>8}'.format(
                indent, name, elapsed, pct))

        # Overhead = total - sum of top-level stages
        top_sum = sum(e for _, e, l in timings if l == 0)
        overhead = total - top_sum
        if overhead > 0.01:
            pct = '{:>4.0f}%'.format(overhead / total * 100)
            self._write('    {:<18}{:>5.1f}s{:>8}'.format(
                'Overhead', overhead, pct))

        self._write('    ' + '\u2500' * 30)
        self._write('    {:<18}{:>5.1f}s'.format('Total', total))

    def _write(self, text):
        print(text, file=self.stream)
