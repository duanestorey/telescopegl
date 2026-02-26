#! /usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of Polymerase.
# Original Telescope code by Matthew L. Bendall (https://github.com/mlbendall/telescope)
#
# New code and modifications by Duane Storey (https://github.com/duanestorey) and Claude (Anthropic).
# Licensed under MIT License.

""" Main functionality of Polymerase

"""
import sys
import os
import argparse
import errno

from polymerase import __version__
from .cli import assign as cli_assign
from .cli import resume as cli_resume
from .cli import plugins as cli_plugins


USAGE = ''' %(prog)s <command> [<args>]

The most commonly used commands are:
   assign         Reassign ambiguous fragments that map to repetitive elements
   resume         Resume previous run from checkpoint file
   list-plugins   List installed primers and cofactors
   install        Install a primer/cofactor package
   test           Generate a command line for testing

'''

def generate_test_command(args):
    _base = os.path.dirname(os.path.abspath(__file__))
    _data_path = os.path.join(_base, 'data')
    _alnpath = os.path.join(_data_path, 'alignment.bam')
    _gtfpath = os.path.join(_data_path, 'annotation.gtf')
    if not os.path.exists(_alnpath):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), _alnpath
        )
    if not os.path.exists(_gtfpath):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), _gtfpath
        )
    print('polymerase assign %s %s' % (_alnpath, _gtfpath), file=sys.stdout)

def main():
    if len(sys.argv) == 1:
        empty_parser = argparse.ArgumentParser(
            description='Tools for analysis of repetitive DNA elements',
            usage=USAGE,
        )
        empty_parser.print_help(sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description='Tools for analysis of repetitive DNA elements',
    )
    parser.add_argument('--version',
        action='version',
        version=__version__,
        default=__version__,
    )

    subparser = parser.add_subparsers(help='Sub-command help', dest='subcommand')

    ''' Parser for bulk RNA-seq assign '''
    assign_parser = subparser.add_parser('assign',
        description='''Reassign ambiguous fragments that map to repetitive elements''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    cli_assign.BulkIDOptions.add_arguments(assign_parser)
    assign_parser.set_defaults(func=lambda args: cli_assign.run(args, sc=False))

    ''' Parser for bulk RNA-seq resume '''
    resume_parser = subparser.add_parser('resume',
        description='''Resume a previous polymerase run''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    cli_resume.BulkResumeOptions.add_arguments(resume_parser)
    resume_parser.set_defaults(func=lambda args: cli_resume.run(args, sc=False))

    ''' Parser for list-plugins '''
    list_plugins_parser = subparser.add_parser('list-plugins',
        description='''List installed primers and cofactors''',
    )
    list_plugins_parser.set_defaults(func=cli_plugins.list_plugins)

    ''' Parser for install '''
    install_parser = subparser.add_parser('install',
        description='''Install a primer/cofactor package''',
    )
    install_parser.add_argument('package', help='Package name to install (e.g. polymerase-primer-chimeric)')
    install_parser.set_defaults(func=cli_plugins.install_plugin)

    ''' Parser for test '''
    test_parser = subparser.add_parser('test',
        description='''Print a test command''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    test_parser.set_defaults(func=lambda args: generate_test_command(args))

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
