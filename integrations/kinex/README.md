# Kinex Harbor integration

Harbor schedules trials and owns retries, timeouts and container cleanup. RoboEnv
provides the simulator, devices, budgets and verifier; Kinex runs agent sessions
and controller tools. Run the commands below from the RoboEnv checkout with its
development environment active. For interactive recording, see
[local Kinex use](../../docs/kinex-local.md).

## Package the agent

Select a Kinex 0.9.0 checkout prepared with `npm ci` and `npm run build`, a Node
executable that satisfies its `engines` field (22.19 or later) and runs in the agent
image (the default main image is Ubuntu 24.04), and a uv executable:

```sh
python tools/package_kinex_agent.py --kinex "$KINEX_REPO" \
  --node "$NODE" --uv "$UV" --out "$ARTIFACTS/kinex-agent.tar.gz"
```

The verified bundle includes the built Kinex application and core package with its
builtins and Python helpers, the production JavaScript dependencies locked by
`package-lock.json`, Node, uv and this integration. It excludes development
dependencies, credentials, user configuration and runs, and does not modify the
Kinex checkout. The bridge is installed at `kinex/integrations/roboenv` and runs
with the bundled Node. It composes Kinex's core runtime (`createKinexRuntime`) and
application (`createApplication`, `runPrintMode`), and imports a few core modules the
runtime does not expose, so it requires the matching Kinex version.

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

Select device names from the resolved rig. The default `integration_probe` reads
`camera`, then sends `arm` one timed joint segment (`write` with `actions` in model
order and `duration_s`). It advances the world clock device (`world wait`; the optional
`world` argument defaults to `world`) until the arm settles near the target and has
moved measurably. It then holds briefly for recording, stops the arm and returns
`completed: true`. It needs a joint arm, an RGB(D) camera and a world clock device.
The synthetic fakeshell world has none of these, so its smoke passes its own tool.

`--agent-config` supplies Harbor agent kwargs. Set attempts, concurrency, retry,
output and timeout in `job.json`; the generated default is one attempt, serial
execution and no retries. Seeds, clocks and native initialization belong to
generated tasks. Regenerate when they change.

The agent prepares `/app/workspace` from public device profiles/drivers. Kinex keeps
the tools' Python as a uv project in `.kinex/python` (`pyproject.toml`, `uv.lock`,
`.venv`); the bridge synchronizes it before execution. Initial dependency resolution
requires network access or a populated cache. The default main image supplies
Python, venv/pip and ffmpeg; custom images need ffmpeg with libx264 and ffv1.
Simulator assets and interpreters belong to the separate native deployment.

Use `tool_directory`, `tool` and `args` for a reusable no-model controller.
`require_completion: true` requires a JSON `completed: true` result without an
error. An empty `recording` disables recording. Controller completion and native
task success are checked independently.

## Development and frozen controllers

Live development requires authorized model use, `mode: "develop"`,
`--model openai/MODEL` and an explicit `effort` (`minimal`, `low`, `medium`, `high`
or `xhigh`) in agent kwargs. Supply credentials through Harbor's agent environment,
outside bundles and tasks.

The bridge runs the Kinex application in-process in headless JSON mode; its event
stream is `agent/kinex.log`. It allows device, Python tool, file and shell
permissions and denies calibration writes; agent questions receive Kinex's headless
reply that no owner answer is available. It checks the actual model, effort and
camera images of the provider requests.

Extract the trial's `agent/workspace.tar.gz` into a new directory; it contains
tools, skills, the Python lock and recordings, excluding the venv.
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
simulator and live-model checks are separate. Run the TypeScript tests and typecheck
from the unpacked bundle, whose Node and dependencies resolve the packaged Kinex:

```sh
cd "$UNPACKED/kinex/integrations/roboenv"
"$UNPACKED/bin/node" --test *.test.ts
"$UNPACKED/bin/node" "$KINEX_REPO/node_modules/typescript/bin/tsc" -p tsconfig.json
```

Preserve failed attempts and their evidence.
