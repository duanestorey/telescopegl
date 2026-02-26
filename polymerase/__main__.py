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


USAGE   = ''' %(prog)s <command> [<args>]

The most commonly used commands are:
   assign    Reassign ambiguous fragments that map to repetitive elements
   resume    Resume previous run from checkpoint file
   test      Generate a command line for testing

'''

def generate_test_command(args):
    try:
        _ = FileNotFoundError()
    except NameError:
        class FileNotFoundError(OSError):
            pass

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
    assign_parser.set_defaults(func=lambda args: cli_assign.run(args, sc = False))

    ''' Parser for bulk RNA-seq resume '''
    resume_parser = subparser.add_parser('resume',
        description='''Resume a previous polymerase run''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    cli_resume.BulkResumeOptions.add_arguments(resume_parser)
    resume_parser.set_defaults(func=lambda args: cli_resume.run(args, sc = False))

    test_parser = subparser.add_parser('test',
        description='''Print a test command''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    test_parser.set_defaults(func=lambda args: generate_test_command(args))

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
