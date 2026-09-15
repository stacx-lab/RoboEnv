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
| Native device bindings | `catalog/bindings/isaaclab/isaaclab-g1.yaml` |
| Copyable robot profiles and drivers | `catalog/robots/isaaclab/isaaclab-g1/{workspace.yaml,devices/}` |
| Runtime model and validation facts | `runtime/src/roboenv_runtime/adapters/isaaclab/g1.json` |
| Pinned asset manifest | `src/roboenv/assets/isaaclab-g1.json` |
| CPU generation receipt | `docs/isaaclab-g1-config.provenance.json` |
| Optional Kinex controller example | `integrations/kinex/g1-motion/` |

The binding plan lists six actuators, `world`, `camera_torso` and `camera_scene`.
All nine have profiles; actuators and cameras also have drivers. Hand postures and
driver resource manifests stay beside the corresponding hand driver. Runtime facts,
asset pins, binding plans and harness tools are not copied with the robot workspace.

The 512×512 RGB-D cameras expose registered optical-Z depth: the torso camera
moves with the waist, while the scene camera is fixed in world coordinates.
They add observation views to the practice interface without altering the upstream
success rule. Camera reads are read-only, cached per control step and never advance
physics. The six-device control snapshot remains separate from image reads.

After preparing the Isaac Lab interpreter and its task dependencies:

```sh
.venv/bin/python tools/prepare_g1_assets.py
.venv/bin/roboenv launch isaaclab/g1-pick-place
```

Asset checksums are stored in `src/roboenv/assets/isaaclab-g1.json`; large USD, texture and
policy files live in `.cache/isaaclab/g1/`. The task uses project-relative paths.
The upstream bundle has some unavailable texture/deleted-reference paths, recorded
in the manifest; geometry and runtime RGB-D output are verified separately.

## Synchronous and realtime operation

Synchronous operation remains the default. To run continuously:

```sh
.venv/bin/roboenv launch isaaclab/g1-pick-place --realtime --port 8124
```

The world owns a main-thread, wall-clock-paced 20 ms control loop. The existing
physics substeps and policy scheduling are unchanged. Device writes replace
retained goals; only the world clock advances physics. A batch replaces all its
goals together, so a tick cannot observe a partially updated batch. This is
best-effort pacing, not hard realtime: inference, rendering or slow requests can
reduce simulation speed. There is no physics skipping or unbounded catch-up.

Realtime base commands expire after 0.5 wall-clock seconds by default. Set
`backend_kwargs.command_timeout_s` in a task YAML to a positive finite duration,
or explicitly set it to null to disable expiry. Only a valid base write refreshes
the deadline; reads, invalid commands and other actuator writes do not. On the
first available tick at/after expiry, travel becomes zero while height, upper-body
targets and the balance policy are retained. This is a software timeout on the
simulation owner thread, not an independent safety watchdog.

Reads acquire current state without requesting a step. Realtime write receipts
acknowledge goal updates and include current state, not post-application feedback.
`sequence` identifies accepted command/stop revisions (including expiry);
`sample_time_s` tracks physics time independently. Use advancing sample times for
fresh feedback. Stop changes the relevant goals without stepping or disabling
balance. Kinex's portable driver stops the device before unloading; the native
unload operation alone is lifecycle-only. Realtime mode disables the native 20 s
timeout and reports an uncapped frame budget (`max: null`), so idle time does not
exhaust the episode. Success and non-timeout termination conditions remain active:
on termination, physics stops advancing, further writes are rejected, and final
observations remain readable. Synchronous mode retains the original horizon.
This does not disable the separate base-command watchdog.

The checked-in robot profile remains stepped. Launch resolves realtime execution
facts for the episode without rewriting the catalog; live exports contain those
resolved facts. Do not copy the static stepped profile into a realtime Kinex run.
Prepare a matching workspace instead (the output directory must be new):

```sh
.venv/bin/python tools/prepare_kinex_workspace.py isaaclab/g1-pick-place \
  --realtime --out /path/to/new-g1-workspace
```

Start Kinex there with `ROBOENV_ENDPOINT=http://127.0.0.1:8124`. Use `--sync` for
both launch and workspace preparation to return to synchronous operation.
Realtime GPU boot, pacing under camera load, balance and replay still require
separate native validation; CPU tests are not evidence of those outcomes.


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
the upstream horizon applies only in synchronous mode.
Upper-body Pink control is replaced by explicit joint targets with gravity
compensation. Reaching a terminal state prevents further motion until reset.

All six devices share `base` as their execution-group coordinator. In synchronous mode, a write
retains every other target and advances 20 ms. `batch_write` validates all targets
before advancing exactly once. `snapshot` and stop do not advance physics. Stops
hold measured arm/hand/waist positions and request zero lower-body velocity at the
retained height. This requests standing; it does not certify balance.

Kinex uses `devices([{device: name, action: "read"}, ...], synchronized=True)`
for consistent reads and the same helper with all-write entries for a coordinated
goal update (one control interval in synchronous mode). Mixed read/write batches are rejected; there is no separate Kinex
`snapshot()` helper or `multi_act` tool. The underlying service still exposes
`base.snapshot` and `base.batch_write`.

The optional `integrations/kinex/g1-motion/` example shows how to combine consistent snapshots,
moving frame conversion, independent arm IK and synchronized writes. It holds the
waist and initial hand posture, follows optional frame-qualified wrist goals, and
uses an explicit iteration budget. In realtime it caps polling at `rate_hz`
(default 20), skips writes on repeated samples, and measures errors from fresh
pre-write feedback rather than treating acknowledgment as applied motion.
The iteration count is not a realtime physics-step count. Existing `servo_to` remains usable for individual
joint motion. A whole-body optimizer is not required by the device interface.
To use the example, copy its directory into the selected Kinex workspace's
`tool/g1-motion/`; the native integration test does this explicitly.
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
