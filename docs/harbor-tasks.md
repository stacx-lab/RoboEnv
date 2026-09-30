# Harbor tasks

Generate standard Harbor tasks from pinned simulator sources or a selected catalog.
Run commands from the RoboEnv checkout with its development environment active.
Commit generators and small examples; keep generated task sets and deployments local.

## Prepare a deployment

Build SDK/runtime wheels with `bash tools/check.sh`, prepare the simulator using
[setup](setup.md), then build its image with `tools/build_simulator_image.py`.
The builder requires `--dependencies`, `--runtime-wheel`, `--out` and `--tag`;
`--interpreter ENV/bin/python` selects the prepared interpreter explicitly.
Otherwise the builder uses the current RoboEnv project’s prepared-installation
registration, as local launch does; `--dependencies` selects asset locations, not
the interpreter. Container defaults live in Harbor deployment code, with optional
`runtime_options` overrides in the generated deployment manifest.

`--interpreter-mode image` embeds the interpreter. `--interpreter-mode depot`
creates an independent read-only copy, verified by image-owned Python before
execution. It does not mount the original environment. Assets can be referenced
or copied with `--asset-mode`. Both image staging and Docker storage need enough
space. Keep original interpreters, assets and run evidence intact.

The builder stages the locked device-service dependencies for the simulator's
Python ABI without installing into its interpreter. A staged build is not a ready
deployment: use the built image's immutable digest and accessible asset depots.
See `tools/build_simulator_image.py --help` for the complete options.

Save a local deployment map, for example `deployments/local/map.json`:

```json
{
  "libero": "../../.cache/images/libero/deployment.json",
  "robocasa": "../../.cache/images/robocasa/deployment.json"
}
```

Relative paths resolve from the map's directory. Simulator packages and assets
must be ready before agent startup. Deployment files contain host-specific paths
and stay outside Git.

## Generate tasks

```sh
roboenv tasks generate --backends libero robocasa \
  --deployments deployments/local/map.json \
  --wheel build/wheels/roboenv-0.9.1-py3-none-any.whl \
  --out generated/benchmark --seeds 0 --check-images
```

The default source is task discovery from pinned upstreams. Use `--ids` to select
public IDs such as `robocasa/pick-place-microwave`, or `--source catalog/tasks` for
checked-in examples. An explicit source directory contains normalized `task.yaml`
files using the [shared task format](simulator-integration.md#task-format).
Omitted `execution_mode` means realtime; an explicit `stepped` setting retains
configured/native horizons. The exported worker uses the same resolved task as
local launch. Each seed produces an independent task.

```sh
roboenv tasks discover --backends libero --out generated/libero-source
roboenv tasks list --source generated/libero-source/tasks
```

Discovery reads source metadata without importing simulator engines. It records
source hashes, revisions, patches and exclusions. Missing task definitions or
required scenes are reported rather than inferred.

Output includes `tasks/`, relative-path `job.json`, `taskset.json` and, when
available, `source-inventory.json`. The generator checks conflicts and inputs
before writing; use a new output directory when existing artifacts differ.
`--check-images` checks local image availability without starting a simulator.

## Task format and references

Every task is validated by the installed Harbor loader against schema 1.4:

```text
task/
  instruction.md
  task.toml
  environment/
    Dockerfile
    docker-compose.yaml
    definition/        Public device facts and portable drivers
    sim/               Owner task parameters and bindings
    sdk/               Selected SDK wheel
  tests/
    test.sh
    verify.py
    task.json
  solution/            Only with explicit reference opt-in
  bundle.json          RoboEnv content hashes
```

The verifier entry is `tests/test.sh`; it writes `/logs/verifier/reward.json`.
`bundle.json` supplements the [Harbor task format](https://www.harborframework.com/docs/tasks).
Generated tasks carry `execution_verified=false` until independently executed.

The checked-in catalog contains no reference solutions. For an explicit source
catalog, solutions are omitted by default; `--include-references` reads
`solution.py` beside an input task. `solution-source.json` can declare `origin`, `official`,
`upstream_url` and `revision`. Without provenance it is a project reference, with
`official=false`. Official attribution requires source and license review.
The generator records the program hash with `replay_verified=false`; a supplied
program or old receipt does not certify a newly generated task.

## Run tasks

```sh
roboenv tasks run generated/benchmark \
  --deployments deployments/local/map.json --agent AGENT --model MODEL
```

The runner supplies local depot paths through environment variables and invokes
`harbor run`. Use an installed Harbor agent. For `--agent kinex --agent-config FILE`,
follow [Kinex packaging and execution](../integrations/kinex/README.md).
Credentials and model endpoints belong in local runtime configuration.

Alternatively, run `harbor run -c job.json -a AGENT -m MODEL` inside the task set
after setting the `ROBOENV_DEPOT_*` variables listed in `taskset.json.required_env`.
`ROBOENV_GPU` selects the GPU. Trials have independent worlds, budgets, identities
and evidence; scheduling and cleanup belong to Harbor.

## Service and observation boundary

The supported deployment uses Linux Docker Compose: `main` hosts the agent and
waits for the `sim` device service at `ROBOENV_ENDPOINT=http://sim:8123`. Separate
trial networks can reuse that internal port. Other providers need their own
support for multiple containers, GPUs and asset mounts.

RoboEnv publishes selected profiles and portable drivers without changing the
static catalog. The Kinex bridge copies them to `/app/workspace/.kinex/`.
Binding plans, scene parameters, owner sockets and verifier evidence remain private.
Sensor-only agents receive task instructions, device facts and sensor readings;
public HTTP and request-status queries must not expose owner results.

The agent uses an unprivileged user; Harbor runs the verifier with owner access
inside `main`. This protects the owner boundary for controlled evaluations, but
is not a separate verifier container or protection from hostile container root.
Read-only mounts still reveal contents: expose only the agent's intended inputs.
New backends must implement and test their public projections explicitly.

Report Harbor format, container boot, device checks, native task success and
fresh-process replay separately. See [simulator validation](gpu-validation.md)
for receipt meanings and [simulator integration](simulator-integration.md) for
backend implementation.

External orchestrators can supply SDK/CLI agents with
[agent-neutral device documentation](device-documentation.md). This export is
independent of Harbor task generation and contains no executable drivers.
