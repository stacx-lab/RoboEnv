#!/bin/sh
set -eu
# Validate every executable depot with image-owned Python before running its code.
if test -d /opt/roboenv/dependency-locks; then
    /usr/bin/python3 -I /opt/roboenv/verify-dependencies.py \
        --manifest-dir /opt/roboenv/dependency-locks
fi
exec "$@"
