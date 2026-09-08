#!/usr/bin/env bash
set -euo pipefail
project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project"
if ! command -v uv >/dev/null 2>&1; then
  echo 'uv is required; install it following https://docs.astral.sh/uv/getting-started/installation/.' >&2
  exit 1
fi
if [[ $# -gt 1 ]]; then
  echo 'Usage: bash tools/bootstrap.sh [VENV_DIRECTORY]' >&2
  exit 2
fi
environment_dir=${1:-.venv}
if [[ -e "$environment_dir" && ! -f "$environment_dir/pyvenv.cfg" ]]; then
  echo 'Refusing an existing target that is not a Python virtual environment.' >&2
  exit 1
fi
# Honor UV_CACHE_DIR, otherwise use uv's shared default cache.
export PYTHONNOUSERSITE=1
if [[ ! -f "$environment_dir/pyvenv.cfg" ]]; then
  uv venv --python 3.12 "$environment_dir"
fi
"$environment_dir/bin/python" -I -c \
  'import sys; assert sys.version_info[:2] == (3, 12), "Development tooling requires Python 3.12"'
uv pip install --python "$environment_dir/bin/python" --no-deps -r locks/dev.txt
uv pip install --python "$environment_dir/bin/python" --no-deps --no-build-isolation -e . -e runtime
uv pip check --python "$environment_dir/bin/python"
echo 'Development environment ready; simulator dependencies are installed separately.'
echo "Activate: source $environment_dir/bin/activate"
echo 'Prepare a simulator: roboenv prepare robotwin'
