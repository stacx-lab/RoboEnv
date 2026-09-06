#!/usr/bin/env python3
"""Single entry point: discover pinned upstream tasks and export standard Harbor tasks."""

import sys

from roboenv.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main(["tasks", "generate", *sys.argv[1:]]))
