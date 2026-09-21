# Kinex Harbor integration

Harbor schedules trials and owns retries, timeouts and container cleanup. RoboEnv
provides the simulator, devices, budgets and verifier; Kinex runs agent sessions
and controller tools. Run the commands below from the RoboEnv checkout with its
development environment active. For interactive recording, see
[local Kinex use](../../docs/kinex-local.md).

## Package the agent

Build the selected Pi-based Kinex checkout with `npm ci && npm run build`.
Select a Node executable (>=22.19 for headless use) and uv:

```sh
python tools/package_kinex_agent.py --kinex "$KINEX_REPO" \
  --node "$NODE" --uv "$UV" --out "$ARTIFACTS/kinex-agent.tar.gz"
```

The version-2 bundle includes compiled Kinex, core assets and Python helpers,
locked JavaScript dependencies, Node, uv and this integration. npm workspace
links are materialized so the extracted bundle can run independently. It excludes credentials,
user configuration and runs, and does not modify the Kinex checkout. The bridge
is installed at `kinex/integrations/roboenv`; its standalone session imports require the matching compiled version. Robotics
operations share the core runtime scope and its awaited cleanup.

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

Kinex runs its native model loop without a step cap. Finite agent kwargs
`max_steps` are rejected; omit the option (legacy `null` is accepted). For a
20-minute execution budget, set `"agent_timeout_sec": 1200`. The execution receipt
records the unlimited model loop and wall-clock deadline. Python preparation precedes
this timer; device activation, recording startup and agent execution count toward it.
The outer Harbor timeout must also allow preparation and artifact collection.
For LIBERO, an unlimited native action budget separately requires
`backend_kwargs.max_steps: null` in the task. Native success, normal agent
completion and execution errors can still end an episode before the deadline.

The agent prepares `/app/workspace` from public device profiles/drivers and creates
a dedicated uv project under `.kinex/python/` (`pyproject.toml`, `uv.lock`,
`.venv`). Ordinary shell environments are not activated or replaced. The bridge synchronizes
its dependencies before execution. Initial dependency resolution
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
tools, skills, the Python lock and recordings, excluding `.kinex/python/.venv`.
Development sessions are retained as `agent/session.jsonl` beside the execution receipt.
Freeze the selected controller using that trial's identity:

```sh
python integrations/kinex/freeze.py "$WORKSPACE" \
  --identity "$TRIAL/agent/bundle-identity.json" \
  --tool solve_task --out "$FROZEN"
```

Set `artifact` to the frozen directory in a new no-model configuration. The
version-2 artifact selects its recorded tool and arguments and freezes the core
Python manifest and lock under `.kinex/python/`. Legacy artifacts must be
revalidated with the new runtime; their existing files are preserved. Bundle identity, Python lock
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
python -m pytest -q tests/test_kinex_node.py tests/test_libero_kinex_python.py
python tools/smoke_device_harbor.py --base-image "$SDK_BASE_IMAGE" \
  --sdk-wheel "$SDK_WHEEL" --runtime-wheel "$RUNTIME_WHEEL" \
  --agent-bundle "$AGENT_BUNDLE" --out "$FRESH_SMOKE_OUT"
```

The last command runs Harbor/Docker/Kinex with a synthetic CPU world. Native
simulator and live-model checks are separate. Run `node --test kinex/integrations/roboenv/*.test.ts` from the unpacked
bundle using its Node executable and dependencies. Imports resolve against the
packaged Kinex runtime. Preserve failed attempts and their evidence.
