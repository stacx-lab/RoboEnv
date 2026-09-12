# Kinex Harbor integration

Harbor schedules trials and owns retries, timeouts and container cleanup. RoboEnv
provides the simulator, devices, budgets and verifier; Kinex runs agent sessions
and controller tools. Run the commands below from the RoboEnv checkout with its
development environment active. For interactive recording, see
[local Kinex use](../../docs/kinex-local.md).

## Package the agent

Select a Kinex checkout with installed Bun dependencies and matching Bun and uv
executables:

```sh
python tools/package_kinex_agent.py --kinex "$KINEX_REPO" \
  --bun "$BUN" --uv "$UV" --out "$ARTIFACTS/kinex-agent.tar.gz"
```

The verified bundle includes Kinex source, builtins, Python helpers, locked
JavaScript dependencies, Bun, uv and this integration. It excludes credentials,
user configuration and runs, and does not modify the Kinex checkout. The bridge
is installed at `kinex/integrations/roboenv`; its imports of Kinex internals require
the matching source version.

## Generate and run tasks

Prepare the SDK wheel and native deployment map using
[Harbor tasks](../../docs/harbor-tasks.md), then:

```sh
roboenv tasks generate --source catalog/tasks --backends robotwin \
  --ids robotwin/stack-blocks-three --seeds 0 7 \
  --deployments "$DEPLOYMENTS" --wheel "$SDK_WHEEL" --out "$TASKSET"
roboenv tasks run "$TASKSET" --deployments "$DEPLOYMENTS" \
  --agent kinex --agent-config "$AGENT_CONFIG"
```

An owner-side configuration for a bounded no-model probe:

```json
{
  "bundle": "/absolute/artifact-path/kinex-agent.tar.gz",
  "mode": "no-model",
  "tool": "integration_probe",
  "args": {"arm": "left_arm", "camera": "head_camera"},
  "recording": {"left_arm": {"rate": 10}, "head_camera": {"rate": 2}}
}
```

Select device names from the resolved rig. `--agent-config` supplies Harbor agent
kwargs. Set attempts, concurrency, retry, output and timeout in `job.json`; the
generated default is one attempt, serial execution and no retries. Seeds, clocks
and native initialization belong to generated tasks. Regenerate when they change.

The agent prepares `/app/workspace` from public device profiles/drivers and creates
a locked Python environment with `kinex python`. Initial dependency resolution
requires network access or a populated cache. The default main image supplies
Python, venv/pip and ffmpeg; custom images need ffmpeg with libx264 and ffv1.
Simulator assets and interpreters belong to the separate native deployment.

Use `tool_directory`, `tool` and `args` for a reusable no-model controller.
`require_completion: true` requires a JSON `completed: true` result without an
error. An empty `recording` disables recording. Controller completion and native
task success are checked independently.

## Development and frozen controllers

Live development requires authorized model use, `mode: "develop"`,
`--model openai/MODEL` and an explicit `effort` in agent kwargs. Supply credentials
through Harbor's agent environment, outside bundles and tasks. The bridge checks
the actual model, effort and camera image requests.

Extract the trial's `agent/workspace.tar.gz` into a new directory; it contains
tools, skills, the Python lock, recordings and sessions, excluding the venv.
Freeze the selected controller using that trial's identity:

```sh
python integrations/kinex/freeze.py "$WORKSPACE" \
  --identity "$TRIAL/agent/bundle-identity.json" \
  --tool solve_task --out "$FROZEN"
```

Set `artifact` to the frozen directory in a new no-model configuration. The
artifact selects its recorded tool and arguments. Bundle identity, Python lock
and device facts must match; content is checked before and after execution.
Each Harbor trial creates a fresh world/workspace. Use its task matrix for repeats
and held-out cases.

## Results and verification

```sh
python tools/report_kinex_harbor.py "$JOB_DIR" --out "$REPORT"
```

Inspect `agent/{execution,integrity,media,bundle-identity}.json`,
`agent/workspace.tar.gz`, `verifier/{submit,reward}.json`,
`verifier/actions.jsonl` and the Harbor trial result. Media receipts apply when
recording was requested. Process exit alone does not certify a trial; verifier
reward remains authoritative even when agent execution fails.

The report separates bridge execution from native success. Fresh-process command
replay requires the matching owner record and backend replay tool; see
[simulator validation](../../docs/gpu-validation.md). Full integration acceptance
is defined in the [testing skill](../../.agents/skills/test-simulator-kinex/SKILL.md).

```sh
bash tools/check.sh
python -m unittest discover -s integrations/kinex -p 'test_*.py'
python tools/smoke_device_harbor.py --base-image "$SDK_BASE_IMAGE" \
  --sdk-wheel "$SDK_WHEEL" --runtime-wheel "$RUNTIME_WHEEL" \
  --agent-bundle "$AGENT_BUNDLE" --out "$FRESH_SMOKE_OUT"
```

The last command runs Harbor/Docker/Kinex with a synthetic CPU world. Native
simulator and live-model checks are separate. Run TypeScript tests/typecheck from
the unpacked bundle with its Bun and dependencies so imports resolve against the
packaged Kinex source. Preserve failed attempts and their evidence.
