# Isaac Lab: G1 (experimental)

Isaac Lab support is experimental. The old Go2 direct-joint posture demo, catalog,
and controller have been removed. No Go2 locomotion policy is supplied. Historical
Go2 provenance receipts are retained as evidence of the former implementation,
not as validation of the current interface.

## G1 loco-manipulation practice interface

G1 follows the catalog ownership rules in [DESIGN.md](../DESIGN.md):

| Content | Canonical location |
| --- | --- |
| Task definition | `catalog/tasks/isaaclab/g1-pick-place/task.yaml` |
| Native device bindings | `src/roboenv/devices/bindings/defaults/isaaclab/isaaclab-g1.yaml` |
| Copyable robot profiles and drivers | `catalog/robots/isaaclab/isaaclab-g1/{workspace.yaml,devices/}` |
| Runtime model and validation facts | `runtime/src/roboenv_runtime/adapters/isaaclab/g1.json` |
| Pinned asset manifest | `src/roboenv/assets/isaaclab-g1.json` |
| CPU generation receipt | `docs/isaaclab-g1-config.provenance.json` |

The binding plan lists six actuators, `world`, `camera_torso` and `camera_scene`.
All nine have profiles; actuators and cameras also have drivers. Hand postures and
driver resource manifests stay beside the corresponding hand driver. Runtime facts,
asset pins, binding plans and harness tools are not copied with the robot workspace.

The 512×512 RGB-D cameras expose registered optical-Z depth: the torso camera
moves with the waist, while the scene camera is fixed in world coordinates.
They add observation views to the practice interface without altering the upstream
success rule. Camera reads are read-only, cached per control step and never advance
physics. Image and actuator reads use individual device calls.

After preparing the Isaac Lab interpreter and its task dependencies:

```sh
.venv/bin/python tools/prepare_g1_assets.py
.venv/bin/roboenv launch isaaclab/g1-pick-place
```

Asset checksums are stored in `src/roboenv/assets/isaaclab-g1.json`; large USD, texture and
policy files live in `.cache/isaaclab/g1/`. The task uses project-relative paths.
The upstream bundle has some unavailable texture/deleted-reference paths, recorded
in the manifest; geometry and runtime RGB-D output are verified separately.

## Execution

Native operations share the simulation creation thread, including rendering and
cleanup. Writes submit absolute action buffers with `duration_s` and return immediately. Realtime advances continuously in 20 ms intervals; stepped advances
while `world.wait` is pending. Velocity expiry zeroes travel and retains posture.
Stops do not advance physics. Native horizons and termination remain active.
See [the execution contract](../DESIGN.md#one-world-one-execution-authority).

```sh
roboenv launch isaaclab/g1-pick-place --port 8124
python tools/prepare_kinex_workspace.py isaaclab/g1-pick-place --out /path/to/new-g1-workspace
```

Start Kinex with `ROBOENV_ENDPOINT=http://127.0.0.1:8124`. Rendering, balance and
native replay need separate validation; CPU tests do not establish those outcomes.

Regenerate the catalog and its runtime facts explicitly with the prepared Isaac Lab
Python (USD bindings required):

```sh
.venv-sim/isaaclab/bin/python tools/generate_g1_config.py \
  --assets .cache/isaaclab/g1/assets.json --out catalog \
  --facts-out runtime/src/roboenv_runtime/adapters/isaaclab/g1.json \
  --provenance docs/isaaclab-g1-config.provenance.json
```

`--out` writes only robots, bindings and tasks beneath that catalog. `--facts-out`
and `--provenance` opt into separate outputs; omit them when checking catalog
reproducibility. The bundled asset manifest is maintained separately from the local
asset cache. Runtime verifies asset and upstream source checksums before starting Kit.

The assembly is `base -> waist -> left_arm/right_arm -> left_hand/right_hand`.
The base declares `control.mode: motion` and `control.modes.locomotion`. It accepts
`{mode: "locomotion", velocity: {vx, vy, wz}, posture: {height}}`; the adapter packs these into the
Agile policy's `[vx,vy,wz,hip_height]` input. Height is ground-relative pelvis height.
Velocity and tracked height are validated before effects. Supplied roll/pitch/yaw
are ignored, because the policy does not track them. Unknown fields and malformed
numbers are rejected. The policy owns leg actuation; no joint override is exposed.
The waist and arms accept their own joint targets and expose local FK/IK. Each TriHand
uses its palm frame with the fixed wrist mount extracted from USD and accepts
seven ordered joint positions; generated `postures.json` contains open,
power and pinch command postures. These are reference configurations, not evidence
of an object grasp. The upstream scene and success condition are retained;
the upstream horizon is retained.
Upper-body Pink control is replaced by explicit joint targets with gravity
compensation. Reaching a terminal state prevents further motion until reset.

Actuators expose read/write/stop and joints optionally expose set. Joint sequences
consume one action period per point; reads and stops do not advance physics. Kinex devices()
can mix concurrent reads and writes, returning individual results in input order.
It does not promise atomic updates or identical acquisition times. The operator
ensures enabled devices do not overlap.

Kinex can control individual arms with `servo` and compose independent device
calls. The native integration check installs `tests/fixtures/device-composition/`
only into its temporary test workspace to exercise frame conversion, arm IK,
concurrent commands and reads. This fixture is not a task-facing tool.
The prepared environment must include the upstream task dependencies, including
`isaaclab_teleop`. Native replay is a separate check from profile/CPU validation.

Rendering capability is determined by a runtime probe. This host's H100 has
successfully produced RGB and depth images; a GPU model name alone is not a
sufficient reason to reject rendering.

Explicit native checks (run outside the CPU pytest suite):

```sh
.venv-sim/isaaclab/bin/python tests/g1_native.py \
  --assets /path/assets.json --out /path/native.json
.venv/bin/python tests/g1_workspace_native.py --catalog /path/catalog \
  --kinex ../kinex --interpreter .venv-sim/isaaclab/bin/python --out /path/fresh-check
```

The second check launches the public service and runs three dual-arm IK/control
iterations through the exported drivers and checks both RGB-D cameras. Its output directory must be new. Neither
check asserts that the steering wheel has been successfully picked and placed.
