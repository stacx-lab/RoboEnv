# Kinex tasks through Harbor

RoboEnv owns task generation, simulator deployments, budgets and verification.
Harbor owns trial scheduling, concurrency, retries, timeouts and container cleanup.
Kinex owns the agent session and normal device/Python tool execution. No simulator
campaign scripts need to be installed in the Kinex checkout.

## Package the agent

Use an explicitly selected Kinex checkout with installed Bun dependencies and a
matching Bun executable. The bundle copies only source, builtins, Python helpers,
locked JavaScript dependencies, Bun, uv and `integrations/kinex`; it excludes Git,
credentials, user configuration, caches and historical runs. It never changes Kinex.

```sh
python tools/package_kinex_agent.py --kinex "$KINEX_REPO" \
  --bun "$BUN" --uv "$UV" --out "$ARTIFACTS/kinex-agent.tar.gz"
```

The bundle is verified before upload. Its bridge is staged under
`kinex/integrations/roboenv` inside the agent container so relative imports use
that exact Kinex source. The source layout in this repository is intentionally
external; run its TypeScript tests/typecheck from the unpacked bundle, not by
resolving imports against RoboEnv's Python `src` directory.

## Generate and schedule tasks

Use the standard exporter and real Harbor 1.4 task loader described in
[harbor-tasks.md](harbor-tasks.md). A task contains `instruction.md`, `task.toml`,
`environment/`, and independent `tests/`; references are explicit opt-in.

```sh
roboenv tasks generate --source catalog/tasks --backends robotwin \
  --ids robotwin/stack-blocks-three --seeds 0 7 \
  --deployments "$DEPLOYMENTS" --wheel "$SDK_WHEEL" --out "$TASKSET"
roboenv tasks run "$TASKSET" --deployments "$DEPLOYMENTS" \
  --agent kinex --agent-config "$AGENT_CONFIG"
```

An owner-side agent configuration for a bounded no-model probe is:

```json
{
  "bundle": "/absolute/artifact-path/kinex-agent.tar.gz",
  "mode": "no-model",
  "tool": "integration_probe",
  "args": {"arm": "left_arm", "camera": "head_camera"},
  "recording": {"left_arm": {"rate": 10}, "head_camera": {"rate": 2}}
}
```

Select actual device names from the resolved robot; cameras differ by rig. Set
`n_attempts`, `n_concurrent_trials`, `retry`, job output and agent timeout in
`job.json` using Harbor's schema. The default generated job has one attempt,
serial trials and no implicit retries. `--agent-config` supplies standard Harbor
agent kwargs; the effective configuration is validated with `JobConfig`. Seeds,
clocks and effective native initialization belong to the generated task, not an
ignored field in agent kwargs. Regenerate tasks when these inputs change.

The Kinex agent creates `/app/workspace`, copies only the published device
profiles/drivers, and prepares a locked Python environment with `kinex python`.
The default main image includes Python with venv/pip and ffmpeg for camera
recording. Custom main images must provide ffmpeg with libx264 and ffv1 encoders.
The agent needs network access for initial Python
dependency resolution unless dependencies are already cached. Native simulator
assets/interpreters belong exclusively to the simulator deployment. The agent
bundle neither installs nor accepts simulator licenses.

Use `tool_directory` plus `tool` and `args` for a reusable no-model controller.
Set `require_completion: true` to require a JSON `completed: true` response with
no error. An empty `recording` explicitly disables recording; such a trial does
not establish the recording acceptance gate. Normal controller completion never
substitutes for the independent verifier's native success predicate.

## Development and frozen controllers

Live development requires explicit authorization, `mode: "develop"`,
`--model openai/MODEL` and an explicit `effort` in agent kwargs. Supply credentials
through Harbor's agent environment configuration, such as `OPENAI_API_KEY`;
never put them in the agent bundle or task. The bridge uses the real Kinex
provider and checks exact model/effort and real camera image requests. This path
is not exercised by the no-model tests.

Harbor's `agent/workspace.tar.gz` preserves tool code, skills, Python lock,
recordings and workspace sessions while excluding the venv. Extract it into a
new directory, then freeze using the trial's recorded identity:

```sh
python integrations/kinex/freeze.py "$WORKSPACE" \
  --identity "$TRIAL/agent/bundle-identity.json" \
  --tool solve_task --out "$FROZEN"
```

Set `artifact` to that frozen directory in a new no-model agent configuration.
The recorded tool/arguments are selected from the artifact. The bundle identity,
Python lock and device facts must match; content is checked before and after
execution. Every Harbor trial supplies a fresh world/workspace. Use Harbor's
trial/task matrix for repeats and held-out cases; do not implement another
scheduler around subprocesses or reuse a live simulator across trials.

## Evidence and limits

Inspect `agent/execution.json`, `agent/integrity.json`, requested
`agent/media.json`, `agent/bundle-identity.json`, `agent/workspace.tar.gz`,
`verifier/submit.json`, `verifier/actions.jsonl`, `verifier/reward.json`, and the
Harbor trial result. A zero process exit without passing receipts fails the
bridge. Harbor reward remains owned by the verifier, including when the agent
fails or returns an optimistic completion message.

Use `python tools/report_kinex_harbor.py JOB_DIR --out REPORT.json` to collect
these independent results. The report distinguishes bridge execution from native
task success. It does not claim live-model capability or deterministic replay.
Native replay is a separate fresh-process check using the matching owner record
and the selected backend replay tool; record identity and tolerances must match.
Full integration acceptance still follows
[the testing skill](../.agents/skills/test-simulator-kinex/SKILL.md).

## Regression commands

```sh
bash tools/check.sh
python -m unittest discover -s integrations/kinex -p 'test_*.py'
python tools/smoke_device_harbor.py --base-image "$SDK_BASE_IMAGE" \
  --sdk-wheel "$SDK_WHEEL" --runtime-wheel "$RUNTIME_WHEEL" \
  --agent-bundle "$AGENT_BUNDLE" --out "$FRESH_SMOKE_OUT"
```

The last command runs a real Harbor/Docker/Kinex trial against a synthetic CPU
fixture and its independent verifier. It is not a physical simulator certificate.
Run native worker conformance/replay independently for the requested simulator,
rig, clock and effective initialization, and preserve failed attempts.
