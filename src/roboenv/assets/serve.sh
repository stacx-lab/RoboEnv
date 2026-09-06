#!/bin/bash
# Transitional deployment: the simulator remains in its existing interpreter.
set -euo pipefail
cd "$ROBOENV_LEGACY_ROOT"
set -a
source ./env.sh
set +a
export TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=/tmp/numba
export MPLCONFIGDIR=/tmp/mpl MUJOCO_GL=egl
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0='*'
export ROBOT_DESCRIPTIONS_CACHE="$ROBOENV_LEGACY_ROOT/.cache/robot_descriptions"
export CAPX_PLANNER_GENEROUS=1 CAPX_PLANNER_SEED=1
export ROBOENV_SOCKET_ROOT=/run ROBOENV_SESSION=harbor ROBOENV_TASKS=/opt/serve/tasks
mkdir -p /run/roboenv /run/roboenv-private
exec "$ROBOENV_INTERPRETER" -m roboenv_runtime.runtime.supervisor \
  --task "$ROBOENV_TASK" --seed "$ROBOENV_SEED" --budget-mode "${ROBOENV_BUDGET_MODE:-official}" \
  --run-dir /run/roboenv/run --session harbor \
  --socket /run/roboenv/sim.sock --worker-socket /run/roboenv-private/worker.sock
