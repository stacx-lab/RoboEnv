#!/usr/bin/env bash
set -euo pipefail
project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project"
python_bin="${ROBOENV_PYTHON:-$project/.venv/bin/python}"
export PYTHONNOUSERSITE=1
export TMPDIR="$project/.cache/tmp"
mkdir -p "$TMPDIR" build/wheels
"$python_bin" -m ruff check src tests tools
"$python_bin" -m ruff check --select E9,F63,F7,F82 runtime/src runtime/packaging
"$python_bin" -m pytest -c pyproject.toml --confcutdir=. tests -q
"$python_bin" -m roboenv.cli.main audit-privacy --project .
# setuptools reuses build/lib in place: a module deleted since the last local build would
# be packaged again, and stale wheels of an older version would sit beside the new ones.
rm -rf build/lib build/bdist.* build/wheels/*.whl runtime/build/lib runtime/build/bdist.*
SOURCE_DATE_EPOCH=1788480000 "$python_bin" -m build --no-isolation --wheel --outdir build/wheels .
SOURCE_DATE_EPOCH=1788480000 "$python_bin" -m build --no-isolation --wheel --outdir build/wheels runtime
"$python_bin" tools/check_wheels.py build/wheels
echo 'CPU/source/packaging gates passed. Simulator replay is a separate gate.'
