# This file is part of Polymerase.
# Original Telescope code by Matthew L. Bendall (https://github.com/mlbendall/telescope)
#
# New code and modifications by Duane Storey (https://github.com/duanestorey) and Claude (Anthropic).
# Licensed under MIT License.

import argparse
import logging
from collections import OrderedDict

import yaml

from .console import Console

# Safe type lookup for YAML-defined CLI options (replaces eval())
_SAFE_TYPES = {
    'int': int,
    'float': float,
    'str': str,
    'bool': bool,
    "argparse.FileType('r')": argparse.FileType('r'),
    "argparse.FileType('w')": argparse.FileType('w'),
}


class SubcommandOptions:
    OPTS = """
    - Input Options:
        - infile:
            positional: True
            help: Input file.
    - Output Options:
        - outfile:
            positional: True
            help: Output file.
    """

    def __init__(self, args):
        self.opt_names, self.opt_groups = self._parse_yaml_opts(self.OPTS)
        for k, v in vars(args).items():
            setattr(self, k, v)

    @classmethod
    def add_arguments(cls, parser):
        opt_names, opt_groups = cls._parse_yaml_opts(cls.OPTS)
        for group_name, args in opt_groups.items():
            argparse_grp = parser.add_argument_group(group_name, '')
            for arg_name, arg_d in args.items():
                _d = dict(arg_d)
                if _d.pop('hide', False):
                    continue
                if _d.pop('positional', False):
                    _arg_name = arg_name
                elif len(arg_name) == 1:
                    _arg_name = f'-{arg_name}'
                else:
                    _arg_name = f'--{arg_name}'

                if 'type' in _d:
                    _type_str = _d['type']
                    if _type_str not in _SAFE_TYPES:
                        raise ValueError(
                            f"Unsupported type '{_type_str}' in CLI option '{arg_name}'. "
                            f'Allowed: {list(_SAFE_TYPES.keys())}'
                        )
                    _d['type'] = _SAFE_TYPES[_type_str]

                argparse_grp.add_argument(_arg_name, **_d)

    @staticmethod
    def _parse_yaml_opts(opts_yaml):
        _opt_names = []
        _opt_groups = OrderedDict()
        for grp in yaml.load(opts_yaml, Loader=yaml.SafeLoader):
            grp_name, args = list(grp.items())[0]
            _opt_groups[grp_name] = OrderedDict()
            for arg in args:
                arg_name, d = list(arg.items())[0]
                _opt_groups[grp_name][arg_name] = d
                _opt_names.append(arg_name)
        return _opt_names, _opt_groups

    def __str__(self):
        ret = []
        if hasattr(self, 'version'):
            ret.append('{:34}{}'.format('Version:', self.version))
        for group_name, args in self.opt_groups.items():
            ret.append(f'{group_name}')
            for arg_name in args:
                v = getattr(self, arg_name, 'Not set')
                v = v.name if hasattr(v, 'name') else v
                ret.append('    {:30}{}'.format(arg_name + ':', v))
        return '\n'.join(ret)


def configure_logging(opts):
    """Configure logging and create Console for structured stdout output.

    Args:
        opts: SubcommandOptions object. Important attributes are "quiet",
              "verbose", "debug", and "logfile".
    Returns:
        Console instance for structured stdout output.
    """
    _quiet = getattr(opts, 'quiet', False)
    _verbose = getattr(opts, 'verbose', False)
    _debug = getattr(opts, 'debug', False)

    # Console level (stdout)
    if _quiet:
        console_level = Console.QUIET
    elif _debug:
        console_level = Console.DEBUG
    elif _verbose:
        console_level = Console.VERBOSE
    else:
        console_level = Console.NORMAL

    # Logging level (stderr) — default is WARNING so normal mode is clean
    if _debug:
        loglev = logging.DEBUG
        logfmt = '%(asctime)s %(levelname)-8s %(message)-60s (%(funcName)s in %(filename)s:%(lineno)d)'
    elif _verbose:
        loglev = logging.INFO
        logfmt = '%(asctime)s %(levelname)-8s %(message)s'
    else:
        loglev = logging.WARNING
        logfmt = '%(asctime)s %(levelname)-8s %(message)s'

    logging.basicConfig(level=loglev, format=logfmt, datefmt='%Y-%m-%d %H:%M:%S', stream=opts.logfile)

    return Console(level=console_level)


# A very large integer
BIG_INT = 2**32 - 1


def backend_display_name(be):
    """Human-readable backend name for console output."""
    names = {
        'gpu': 'GPU (CuPy)',
        'cpu_optimized': 'CPU-Optimized (Numba)',
        'cpu_stock': 'CPU (scipy)',
    }
    return names.get(be.name, be.name)


def collect_output_files(outdir):
    """Collect output files relative to outdir, sorted with top-level first."""
    import os

    results = []
    for root, _dirs, files in os.walk(outdir):
        for f in sorted(files):
            relpath = os.path.relpath(os.path.join(root, f), outdir)
            results.append(relpath)
    # Sort: top-level files first, then subdirectory files
    results.sort(key=lambda p: (os.sep in p, p))
    return results
