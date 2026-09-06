#!/bin/bash
set -euo pipefail
export ROBOENV_ROOT=/opt/roboenv/runtime
export ROBOENV_TASKS=/opt/serve/tasks ROBOENV_SESSION=harbor ROBOENV_SOCKET_ROOT=/run
export ROBOENV_SOCKET=/run/roboenv/sim.sock ROBOENV_GENERATION=0
export TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=/tmp/numba
export MPLCONFIGDIR=/tmp/mpl MUJOCO_GL=egl XDG_CACHE_HOME=/tmp/cache
export ROBOT_DESCRIPTIONS_CACHE=/opt/roboenv/runtime/.cache/robot_descriptions
export CAPX_PLANNER_GENEROUS=1 CAPX_PLANNER_SEED=1
# The pure-Python SDK wheel is mounted read-only; no network/install at startup.
export PYTHONPATH="${ROBOENV_SDK_PATH:?SDK wheel required}:${PYTHONPATH:-}"
mkdir -p /run/roboenv /run/roboenv-private
if test -d /opt/roboenv/dependency-locks; then
  /usr/bin/python3 -I /opt/roboenv/verify-dependencies.py \
    --manifest-dir /opt/roboenv/dependency-locks
fi
exec /opt/sim-env/bin/python -m roboenv_runtime.runtime.supervisor \
  --task "$ROBOENV_TASK" --seed "$ROBOENV_SEED" --budget-mode "${ROBOENV_BUDGET_MODE:-official}" \
  --run-dir /run/roboenv/run --session harbor \
  --socket /run/roboenv/sim.sock --worker-socket /run/roboenv-private/worker.sock
