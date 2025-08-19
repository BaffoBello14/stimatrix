#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--unit', action='store_true')
    parser.add_argument('--integration', action='store_true')
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--slow', action='store_true')
    parser.add_argument('--coverage', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args, extra = parser.parse_known_args()

    cmd = [sys.executable, '-m', 'pytest']
    if args.unit:
        cmd += ['-m', 'unit']
    if args.integration:
        cmd += ['-m', 'integration']
    if args.fast:
        cmd += ['-m', 'not slow']
    if args.slow:
        cmd += ['-m', 'slow']
    if args.coverage:
        cmd += ['--cov=src', '--cov-report=html:htmlcov']
    if args.verbose:
        cmd += ['-v']
    cmd += extra
    return subprocess.call(cmd)


if __name__ == '__main__':
    raise SystemExit(main())

